import torch
import torch.nn as nn
import torch.nn.functional as F
from fungi_network import tokenize_attributes, AttributeEmbedder, ensure_folder, seed_torch, initialize_csv_logger, log_epoch_to_csv, \
    get_transforms, FungiDataset
import wandb
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import ViTModel
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv 

from attributeEmbedder_v2 import AttributeEmbedderV2
from focal_loss_fun import FocalLoss

# ----------------------------------------
# Helpers: ImageNet->ViT normalization hop
# ----------------------------------------
class ImagenetToViTNorm(nn.Module):
    """
    Converts tensors normalized with ImageNet stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    to the default ViT normalization (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]).
    Works on (B,3,H,W) tensors.
    """
    def __init__(self):
        super().__init__()
        im_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        im_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        vit_mean = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)
        vit_std  = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)

        # x_im = (x - im_mean)/im_std  ->  want x_vit = (x - vit_mean)/vit_std
        # substitute x = x_im*im_std + im_mean
        # x_vit = ((x_im*im_std + im_mean) - vit_mean)/vit_std
        self.register_buffer("shift", (im_mean - vit_mean) / vit_std)
        self.register_buffer("scale", im_std / vit_std)

    def forward(self, x_imnorm: torch.Tensor) -> torch.Tensor:
        return x_imnorm * self.scale + self.shift


# ----------------------------------------
# Transformer bits for fusion
# ----------------------------------------
class CrossAttentionBlock(nn.Module):
    """
    Updates the attribute token stream (queries) using:
      optional self-attention over attributes + cross-attention over ViT tokens.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1, self_attn: bool = True):
        super().__init__()
        self.self_attn = self_attn

        if self_attn:
            self.sa_ln = nn.LayerNorm(d_model)
            self.sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ca_ln = nn.LayerNorm(d_model)
        self.ca = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )

    def forward(self, attr_tokens: torch.Tensor, vit_tokens: torch.Tensor, attr_mask: torch.Tensor | None = None):
        """
        attr_tokens: (B, T_attr, D)
        vit_tokens:  (B, T_vit, D)
        attr_mask:   (B, T_attr) 1=keep, 0=pad (optional)
        """
        x = attr_tokens

        if self.self_attn:
            y = self.sa_ln(x)
            y, _ = self.sa(y, y, y,
                           key_padding_mask=(attr_mask == 0) if attr_mask is not None else None,
                           need_weights=False)
            x = x + y

        y = self.ca_ln(x)
        y, _ = self.ca(y, vit_tokens, vit_tokens, need_weights=False)
        x = x + y

        y = self.ffn_ln(x)
        y = self.ffn(y)
        x = x + y
        return x


class MultiModalViTFusion(nn.Module):
    """
    Frozen ViT backbone + transformer fusion head that combines attribute tokens with ViT tokens.
    Plugs into your existing training/eval loops with the same call signature as MultiModalEffNet.
    """
    def __init__(
        self,
        num_classes: int,
        attr_embedder,                 # your AttributeEmbedder instance (kept as-is)
        attr_token_dim: int = 64,      # E from your embedder (num_embedding_dims)
        attr_num_tokens: int = 6,      # [habitat, substrate, month, hour, camera_model, camera_maker, geo]
        vit_name: str = "google/vit-base-patch16-224",
        cross_attn_heads: int = 8,
        cross_attn_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pool: str = "cls+vitcls",      # "cls", "mean", "cls+vitcls"
        use_attr_self_attn: bool = True,
    ):
        super().__init__()
        self.attr = attr_embedder
        self.attr_token_dim = attr_token_dim
        self.attr_num_tokens = attr_num_tokens
        self.pool = pool

        # 1) Frozen ViT
        self.vit = ViTModel.from_pretrained(vit_name)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        self.vit_hidden = self.vit.config.hidden_size  # usually 768

        # Adapter to convert your ImageNet-normalized tensors to ViT normalization
        self.im2vit = ImagenetToViTNorm()

        # 2) Attribute stream: (B, 7*E) -> (B, 7, E) -> project to D
        #self.attr_proj = nn.Linear(attr_token_dim, self.vit_hidden, bias=True)
        self.attr_proj = nn.Sequential(nn.Linear(attr_token_dim, self.vit_hidden//2),
                                       nn.GELU(),
                                       nn.Linear(self.vit_hidden//2, self.vit_hidden),
                                       nn.LayerNorm(self.vit_hidden))

        # Learnable [NI-CLS] for attributes
        self.attr_cls = nn.Parameter(torch.zeros(1, 1, self.vit_hidden))
        nn.init.trunc_normal_(self.attr_cls, std=0.02)

        # Positional embeddings for attributes (7 tokens + CLS)
        self.attr_pos = nn.Parameter(torch.zeros(1, attr_num_tokens + 1, self.vit_hidden))
        nn.init.trunc_normal_(self.attr_pos, std=0.02)

        # 3) Cross-attention stack
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=self.vit_hidden,
                n_heads=cross_attn_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                self_attn=use_attr_self_attn,
            ) for _ in range(cross_attn_layers)
        ])
        self.final_ln = nn.LayerNorm(self.vit_hidden)

        # 4) Classifier
        cls_in = self.vit_hidden * (2 if pool == "cls+vitcls" else 1)
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, self.vit_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.vit_hidden, num_classes),
        )

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor):
        out = self.vit(pixel_values=pixel_values, output_hidden_states=False)
        vit_tokens = out.last_hidden_state         # (B, T_vit, D) includes [CLS] at idx 0
        vit_cls = vit_tokens[:, 0]                # (B, D)
        return vit_tokens, vit_cls

    def _attributes_to_tokens(self, attr_vec: torch.Tensor) -> torch.Tensor:
        """
        attr_vec: (B, 7*E) from your AttributeEmbedder
        returns: (B, 7, D) tokens projected to ViT hidden size
        """
        B, FEAT = attr_vec.shape
        E = self.attr_token_dim
        T = self.attr_num_tokens
        assert FEAT == E * T, f"Expected attr dim {E*T}, got {FEAT}"
        tokens_E = attr_vec.view(B, T, E)             # (B, 7, 64)
        tokens_D = self.attr_proj(tokens_E)           # (B, 7, D)
        return tokens_D

    def forward(self, images, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude):
        B = images.size(0)

        # 1) Image path
        # Convert ImageNet-normalized tensors to ViT’s expected norm
        vit_pixels = self.im2vit(images)
        with torch.no_grad():
            vit_tokens, vit_cls = self.encode_image(vit_pixels)   # (B, T_vit, D), (B, D)

        # 2) Attribute path
        attr_vec = self.attr(habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude)  # (B, 7*E)
        attr_tokens = self._attributes_to_tokens(attr_vec)         # (B, 7, D)

        # Prepend [NI-CLS] and add learned positions
        ni_cls = self.attr_cls.expand(B, -1, -1)                  # (B,1,D)
        x = torch.cat([ni_cls, attr_tokens], dim=1)               # (B, 8, D)
        x = x + self.attr_pos[:, :x.size(1), :]                   # (B, 8, D)

        # (Optional) no padding in your attributes (always 7 tokens), so mask is None
        for blk in self.blocks:
            x = blk(x, vit_tokens, attr_mask=None)

        x = self.final_ln(x)
        ni_cls_pooled = x[:, 0]                                   # (B, D)

        if self.pool == "cls":
            fused = ni_cls_pooled
        elif self.pool == "mean":
            fused = x.mean(dim=1)
        elif self.pool == "cls+vitcls":
            fused = torch.cat([ni_cls_pooled, vit_cls], dim=-1)
        else:
            raise ValueError(f"Unknown pool mode: {self.pool}")

        logits = self.classifier(fused)                            # (B, num_classes)
        return logits
    


def train_fungi_network(data_file, image_path, checkpoint_dir, multi_modal=False, wandb_bool=False):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    """
    # Ensure checkpoint directory exists
    CameraModelSTxt = '/home/awias/data/Summerschool_2025/metadata_1/camera_models.txt'
    CameraMakerTxt = '/home/awias/data/Summerschool_2025/metadata_1/camera_makers.txt'

    ensure_folder(checkpoint_dir)

    # Set Logger
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    # Load metadata
    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    # Initialize DataLoaders
    train_dataset = FungiDataset(train_df, image_path, CameraModelSTxt, CameraMakerTxt, transform=get_transforms(data='train'), multi_modal=multi_modal)
    valid_dataset = FungiDataset(val_df, image_path, CameraModelSTxt, CameraMakerTxt,  transform=get_transforms(data='valid'), multi_modal=multi_modal)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=8)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = train_df['taxonID_index'].nunique()
    # csv_embedder = AttributeEmbedder(
    #     num_habitats=train_dataset.tokenizer.num_habitats,
    #     num_substrates=train_dataset.tokenizer.num_substrates,
    #     num_months=train_dataset.tokenizer.num_months,
    #     num_hours=train_dataset.tokenizer.num_hours,
    #     num_camera_models=train_dataset.tokenizer.num_camera_models,
    #     num_camera_makers=train_dataset.tokenizer.num_camera_makers,
    #     num_embedding_dims=64
    # )
    csv_embedder = AttributeEmbedderV2(
        num_habitats=train_dataset.tokenizer.num_habitats,
        num_substrates=train_dataset.tokenizer.num_substrates,
        num_camera_models=train_dataset.tokenizer.num_camera_models,
        num_camera_makers=train_dataset.tokenizer.num_camera_makers,
        num_embedding_dims=64,  # E
        fourier_features=8,     # F
        geo_scale=10.0          # scale for Fourier features
    )
    model = MultiModalViTFusion(
        num_classes=num_classes,
        attr_embedder=csv_embedder,
        attr_token_dim=64,
        attr_num_tokens=6,
        vit_name="google/vit-base-patch16-224",
        cross_attn_heads=8,
        cross_attn_layers=4,
        mlp_ratio=4.0,
        dropout=0.1,
        pool="cls+vitcls",
        use_attr_self_attn=True,
    ).to(device)

    # IMPORTANT: only train unfrozen params (fusion head)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=5e-4)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, eps=1e-6)
    #criterion = nn.CrossEntropyLoss()
    counts = train_df['taxonID_index'].value_counts().sort_index()  # index 0..(C-1)
    beta = 0.9999  # tune in [0.9, 0.9999]; closer to 1 = stronger reweighting
    eff_num = 1.0 - np.power(beta, counts.values)
    class_weights = (1.0 - beta) / eff_num
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize around 1

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    criterion = FocalLoss(gamma=2.0, weight=class_weights.to(device), label_smoothing=0.05)

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0
    if wandb_bool:
        wandb.init(project="sc2025", entity="Bjonze", name="vit-base-patch16-224")

    # Training Loop
    global_step = 0
    running_accuracy = 0
    for epoch in range(100):  # Maximum epochs
        model.train()
        train_loss = 0.0
        total_correct_train = 0
        total_train_samples = 0
        
        # Start epoch timer
        epoch_start_time = time.time()
        
        # Training Loop
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(desc=f"Epoch {epoch + 1}/{100} - Training")
        
        if multi_modal:
            for step, batch in progress_bar:
                images, labels, _, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude = batch
                images = images.to(device); labels = labels.to(device).long()
                habitat = habitat.to(device).long(); substrate = substrate.to(device).long()
                month = month.to(device).long()
                day = day.to(device).long()
                camera_model = camera_model.to(device).long(); camera_maker = camera_maker.to(device).long()
                latitude = latitude.to(device).float(); longitude = longitude.to(device).float()

                optimizer.zero_grad()
                outputs = model(images, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate train accuracy
                running_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0) 
                total_correct_train += (outputs.argmax(1) == labels).sum().item()
                total_train_samples += labels.size(0)
                global_step += 1
                log_dict = {
                    "train_loss": loss.item(),
                    "running_train_accuracy": running_accuracy / global_step,
                }
                if wandb_bool:
                    wandb.log(log_dict)
        else:
            for step, batch in progress_bar:
                images, labels, _ = batch
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate train accuracy
                total_correct_train += (outputs.argmax(1) == labels).sum().item()
                total_train_samples += labels.size(0)
                running_accuracy += (outputs.argmax(1) == labels).sum().item() / labels.size(0) 
                global_step += 1
                log_dict = {
                    "train_loss": loss.item(),
                    "running_train_accuracy": running_accuracy / global_step,
                }
                if wandb_bool:
                    wandb.log(log_dict)
        
        # Calculate overall train accuracy and average loss
        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        total_correct_val = 0
        total_val_samples = 0
        
        # Validation Loop
        progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
        progress_bar.set_description(desc=f"Validation epoch {epoch + 1}/{100}")
        with torch.no_grad():
            if multi_modal:
                for step, batch in progress_bar:
                    images, labels, _, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude = batch
                    images = images.to(device); labels = labels.to(device).long()
                    habitat = habitat.to(device).long(); substrate = substrate.to(device).long()
                    month = month.to(device).long(); day = day.to(device).long()
                    camera_model = camera_model.to(device).long(); camera_maker = camera_maker.to(device).long()
                    latitude = latitude.to(device).float(); longitude = longitude.to(device).float()

                    optimizer.zero_grad()
                    outputs = model(images, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude)
                    
                    val_loss += criterion(outputs, labels).item()
                    
                    # Calculate validation accuracy
                    total_correct_val += (outputs.argmax(1) == labels).sum().item()
                    total_val_samples += labels.size(0)
                
            else:
                for step, batch in progress_bar:
                    images, labels, _ = batch
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item()
                    
                    # Calculate validation accuracy
                    total_correct_val += (outputs.argmax(1) == labels).sum().item()
                    total_val_samples += labels.size(0)

        # Calculate overall validation accuracy and average loss
        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)
        scheduler.step(avg_val_loss)

        # (optional) log/print the current LR
        current_lr = optimizer.param_groups[0]["lr"]        
        log_dict = {
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch + 1,
            "lr": current_lr
        }
        if wandb_bool:
            wandb.log(log_dict)

        # Stop epoch timer and calculate elapsed time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Print summary at the end of the epoch
        print(f"Epoch {epoch + 1} Summary: "
            f"Train Loss = {avg_train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, "
            f"Val Loss = {avg_val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}, "
            f"Epoch Time = {epoch_time:.2f} seconds")
        
        # Log epoch metrics to the CSV file
        log_epoch_to_csv(csv_file_path, epoch + 1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        # Save Models Based on Accuracy and Loss
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"Epoch {epoch + 1}: Best accuracy updated to {best_accuracy:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"Epoch {epoch + 1}: Best loss updated to {best_loss:.4f}")
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
            break

def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name, multi_modal=False):
    """
    Evaluate network on the test set and save predictions to a CSV file.
    """
    # Ensure checkpoint directory exists
    CameraModelSTxt = 'C:/Users/bmsha/sc2025/metadata_1/camera_models.txt'
    CameraMakerTxt = 'C:/Users/bmsha/sc2025/metadata_1/camera_makers.txt'
    
    ensure_folder(checkpoint_dir)

    # Model and Test Setup
    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]
    test_dataset = FungiDataset(test_df, image_path, CameraModelSTxt, CameraMakerTxt, transform=get_transforms(data='valid'), multi_modal=multi_modal)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 183 #test_df['taxonID_index'].nunique()
    

    # csv_embedder = AttributeEmbedder(
    #     num_habitats=test_dataset.tokenizer.num_habitats,
    #     num_substrates=test_dataset.tokenizer.num_substrates,
    #     num_months=test_dataset.tokenizer.num_months,
    #     num_hours=test_dataset.tokenizer.num_hours,
    #     num_camera_models=test_dataset.tokenizer.num_camera_models,
    #     num_camera_makers=test_dataset.tokenizer.num_camera_makers,
    #     num_embedding_dims=64
    # )
    csv_embedder = AttributeEmbedderV2(
        num_habitats=test_dataset.tokenizer.num_habitats,
        num_substrates=test_dataset.tokenizer.num_substrates,
        num_camera_models=test_dataset.tokenizer.num_camera_models,
        num_camera_makers=test_dataset.tokenizer.num_camera_makers,
        num_embedding_dims=64,  # E
        fourier_features=8,     # F
        geo_scale=10.0          # scale for Fourier features
    )
    model = MultiModalViTFusion(
        num_classes=num_classes,
        attr_embedder=csv_embedder,
        attr_token_dim=64,
        attr_num_tokens=6,
        vit_name="google/vit-base-patch16-224",
        cross_attn_heads=8,
        cross_attn_layers=4,
        mlp_ratio=4.0,
        dropout=0.1,
        pool="cls+vitcls",
        use_attr_self_attn=True,
    ).to(device)
    
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)

    # Collect Predictions
    results = []
    model.eval()
    
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
    progress_bar.set_description(desc=f"Final Evaluation on Test Set - {session_name}")
        
    with torch.no_grad():
        if multi_modal:
            for step, batch in progress_bar:
                # Unpack batch
                images, _, filenames, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude = batch
                images = images.to(device)
                habitat = habitat.to(device).long(); substrate = substrate.to(device).long()
                month = month.to(device).long(); day = day.to(device).long()
                camera_model = camera_model.to(device).long(); camera_maker = camera_maker.to(device).long()
                latitude = latitude.to(device).float(); longitude = longitude.to(device).float()

                outputs = model(images, habitat, substrate, month, day, camera_model, camera_maker, latitude, longitude)
                predictions = outputs.argmax(1).cpu().numpy()
                results.extend(zip(filenames, predictions))
        else:
            for step, batch in progress_bar:
                images, _, filenames = batch
                images = images.to(device)
                outputs = model(images).argmax(1).cpu().numpy()
                results.extend(zip(filenames, outputs))  # Store filenames and predictions only

    # Save Results to CSV
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])  # Write session name as the first line
        writer.writerows(results)  # Write filenames and predictions
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    # Path to fungi images
    image_path = "/home/awias/data/Summerschool_2025/FungiImages"
    # Path to metadata file
    data_file = '/home/awias/data/Summerschool_2025/metadata_1/metadata_fused.csv'

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = "firsttry"
    wandb_bool = True
    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"/home/awias/data/Summerschool_2025/checkpoints/{session}/")
    seed_torch(42)  # Set random seed for reproducibility
    train_fungi_network(data_file, image_path, checkpoint_dir, multi_modal=True, wandb_bool=wandb_bool)
    wandb.finish() if wandb_bool else None
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session, multi_modal=True)