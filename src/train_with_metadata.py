import os
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from albumentations import Compose, Normalize, Resize
from albumentations import RandomResizedCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torchvision import models
import tqdm
import numpy as np
from PIL import Image
import time
import csv

# -------------------------
# Utility functions
# -------------------------
def ensure_folder(folder):
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def initialize_csv_logger(file_path):
    header = ["epoch", "time", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy])

def get_transforms(data):
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop(size=(height, width), scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        raise ValueError("Unknown mode (train/valid)")

def create_categorical_tokenizers(df, categorical_keys):
    """Create a mapping from category value to integer for each categorical column, reserving the last index for NaN."""
    tokenizers = {}
    for key in categorical_keys:
        unique_vals = sorted(df[key].dropna().unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        nan_token = len(mapping)  # Reserve the last index for NaN
        mapping["__nan__"] = nan_token
        tokenizers[key] = mapping
    return tokenizers

def apply_tokenizers(df, tokenizers):
    """Apply the categorical tokenizers to the DataFrame in-place, mapping NaN to the special token."""
    for key, mapping in tokenizers.items():
        nan_token = mapping["__nan__"]
        df[key] = df[key].map(lambda x: mapping.get(x, nan_token))
    return df

# -------------------------
# Dataset
# -------------------------
class FungiDataset(Dataset):
    def __init__(self, df, path, categorical_keys=None, continuous_keys=None, transform=None):
        self.df = df
        self.path = path
        self.transform = transform
        self.categorical_keys = categorical_keys if categorical_keys else []
        self.continuous_keys = continuous_keys if continuous_keys else []

        # For normalization of continuous features
        self.cont_stats = {}
        for k in self.continuous_keys:
            self.cont_stats[k] = {
                "mean": df[k].mean(),
                "std": df[k].std() if df[k].std() > 0 else 1.0
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['filename_index']
        label = row['taxonID_index']
        label = -1 if pd.isnull(label) else int(label)

        with Image.open(os.path.join(self.path, file_path)) as img:
            image = img.convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        # Categorical: as long tensors
        cat_extras = [torch.tensor(row[k], dtype=torch.long) for k in self.categorical_keys]
        # Continuous: as float tensors, normalized
        cont_extras = [
            torch.tensor(
                (row[k] - self.cont_stats[k]["mean"]) / self.cont_stats[k]["std"],
                dtype=torch.float
            ).unsqueeze(0)
            for k in self.continuous_keys
        ]

        return image, label, file_path, cat_extras, cont_extras

# -------------------------
# Model
# -------------------------
class MultiModalEfficientNet(nn.Module):
    def __init__(self, num_classes, categorical_cardinalities, num_continuous, embed_dim=32, mlp_hidden=64):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        # self.backbone.classifier = nn.Identity()
        # load weights from path
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 183)
        )
        self.load_weights("/home/mmilo/FungiChallenge/results/EfficientNet_20250812_091422/best_accuracy.pth")
        self.backbone.classifier = nn.Identity()  # Remove the final classifier for custom head
        # Freeze the backbone to prevent training  
        for param in self.backbone.parameters():
            param.requires_grad = False      

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=embed_dim)
            for card in categorical_cardinalities
        ])
        self.num_continuous = num_continuous

        # MLP for metadata
        meta_in_dim = embed_dim * len(categorical_cardinalities) + num_continuous
        self.mlp = nn.Sequential(
            nn.Linear(meta_in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features + mlp_hidden, in_features + mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features + mlp_hidden, num_classes)
        )

    def forward(self, images, cat_extras, cont_extras):
        img_feats = self.backbone(images)
        if self.embeddings:
            embeds = [emb(extra) for emb, extra in zip(self.embeddings, cat_extras)]
            embeds = torch.cat(embeds, dim=1) if embeds else torch.empty((images.size(0), 0), device=images.device)
        else:
            embeds = torch.empty((images.size(0), 0), device=images.device)
        if cont_extras:
            cont_feats = torch.cat(cont_extras, dim=1)
        else:
            cont_feats = torch.empty((images.size(0), 0), device=images.device)
        meta_feats = torch.cat([embeds, cont_feats], dim=1)
        mlp_feats = self.mlp(meta_feats)
        combined = torch.cat([img_feats, mlp_feats], dim=1)
        return self.classifier(combined)
    
    def load_weights(self, path=None):
        """Load weights from a given path or use default pretrained weights."""
        if path and os.path.exists(path):
            self.backbone.load_state_dict(torch.load(path))
            print(f"Loaded weights from {path}")
        else:
            print("Using default pretrained weights for EfficientNet-B0.")

import torch
import torch.nn as nn
import timm  # for pretrained ViT
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', num_classes=None, task_type='multi-class'):
        """
        Focal Loss for multi-class or binary classification.
        Args:
            gamma (float): focusing parameter.
            alpha (list, float, or None): class weights. If None, no weighting.
            reduction (str): 'mean', 'sum', or 'none'.
            num_classes (int): number of classes (required for multi-class).
            task_type (str): 'multi-class' or 'binary'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        if alpha is not None:
            if isinstance(alpha, (list, torch.Tensor)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, input, target):
        if self.task_type == 'multi-class':
            # input: (batch, num_classes), target: (batch,)
            logpt = F.log_softmax(input, dim=1)
            pt = torch.exp(logpt)
            logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
            pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
            if self.alpha is not None:
                at = self.alpha.to(input.device)[target]
                logpt = logpt * at
            loss = -((1 - pt) ** self.gamma) * logpt
        elif self.task_type == 'binary':
            # input: (batch,), target: (batch,)
            logpt = F.logsigmoid(input)
            pt = torch.exp(logpt)
            if self.alpha is not None:
                at = self.alpha.to(input.device)
                logpt = logpt * at
            loss = -((1 - pt) ** self.gamma) * logpt
        else:
            raise ValueError("Unknown task_type: choose 'multi-class' or 'binary'.")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class SimpleMultiModalFusion(nn.Module):
    def __init__(self, num_classes, categorical_cardinalities, num_continuous,
                 embed_dim=32, mlp_hidden=128, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained ViT base (patch16_224)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.reset_classifier(0)  # Remove classification head
        
        
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=embed_dim)
            for card in categorical_cardinalities
        ])
        
        self.num_continuous = num_continuous
        
        # Calculate input dim for metadata MLP
        meta_in_dim = embed_dim * len(categorical_cardinalities) + num_continuous
        
        # Metadata MLP
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        vit_embed_dim = self.vit.embed_dim  # Typically 768 for vit_base_patch16_224
        
        # Final classifier head with layer normalization
        self.classifier = nn.Sequential(
            nn.LayerNorm(vit_embed_dim + mlp_hidden),
            nn.Linear(vit_embed_dim + mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, num_classes)
        )
        
        self.load_vit_model()  # Load weights if available
        
        
    def forward(self, images, cat_extras, cont_extras):
        B = images.size(0)
        
        # Extract ViT features from CLS token
        vit_tokens = self.vit.forward_features(images)  # (B, num_patches+1, D)
        cls_token = vit_tokens[:, 0, :]  # CLS token (B, D)
        
        # Embed categorical features and concatenate
        if self.embeddings:
            embeds = [emb(extra) for emb, extra in zip(self.embeddings, cat_extras)]
            embeds = torch.cat(embeds, dim=1) if embeds else torch.empty((B, 0), device=images.device)
        else:
            embeds = torch.empty((B, 0), device=images.device)
        
        # Concatenate continuous features
        if cont_extras:
            cont_feats = torch.cat(cont_extras, dim=1)
        else:
            cont_feats = torch.empty((B, 0), device=images.device)
        
        meta_feats = torch.cat([embeds, cont_feats], dim=1)
        meta_feats = self.meta_mlp(meta_feats)
        
        # Concatenate ViT CLS features and metadata features
        combined = torch.cat([cls_token, meta_feats], dim=1)
        
        # Classification
        return self.classifier(combined)
    
    def load_vit_model(self, path="/home/mmilo/FungiChallenge/results/the_bear_no_metadata/vit_weights.pth"):
        """Load weights from a given path or use default pretrained weights."""
        if path and os.path.exists(path):
            self.vit.load_state_dict(torch.load(path))
            print(f"Loaded weights from {path}")
        else:
            print("Using default pretrained weights for ViT.")


# -------------------------
# Training function
# -------------------------
def train_fungi_network(data_file, 
                        image_path, 
                        checkpoint_dir, 
                        categorical_keys, 
                        continuous_keys, 
                        tokenizers=None, 
                        drop_na=True,
                        n_epochs=100):
    ensure_folder(checkpoint_dir)
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    meta_keys = (categorical_keys if categorical_keys else []) + (continuous_keys if continuous_keys else [])

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['taxonID_index'])

    if drop_na and meta_keys:
        train_df = train_df.dropna(subset=meta_keys)
        val_df = val_df.dropna(subset=meta_keys)
    print('Training size:', len(train_df))
    print('Validation size:', len(val_df))

    if categorical_keys:
        if tokenizers is None:
            tokenizers = create_categorical_tokenizers(df, categorical_keys)
        train_df = apply_tokenizers(train_df, tokenizers)
        val_df = apply_tokenizers(val_df, tokenizers)

    categorical_cardinalities = [len(tokenizers[k]) if k in tokenizers else 0 for k in categorical_keys]
    num_continuous = len(continuous_keys)

    train_dataset = FungiDataset(train_df, image_path, categorical_keys=categorical_keys, continuous_keys=continuous_keys, transform=get_transforms('train'))
    valid_dataset = FungiDataset(val_df, image_path, categorical_keys=categorical_keys, continuous_keys=continuous_keys, transform=get_transforms('valid'))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=32, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMultiModalFusion(
        num_classes=183,
        categorical_cardinalities=categorical_cardinalities,
        num_continuous=num_continuous,
        freeze_backbone=True
    ).to(device)



    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True, eps=1e-6)
    
    criterion = FocalLoss(gamma=2, task_type='multi-class', num_classes=183)
    criterion = nn.CrossEntropyLoss()

    patience = 20
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        train_loss, total_correct_train, total_train_samples = 0.0, 0, 0
        epoch_start_time = time.time()

        for images, labels, _, cat_extras, cont_extras in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            cat_extras = [e.to(device) for e in cat_extras]
            cont_extras = [e.to(device) for e in cont_extras]

            optimizer.zero_grad()
            outputs = model(images, cat_extras, cont_extras)

            if outputs.shape[0] != labels.shape[0]:
                print("Warning: output and label batch size mismatch", outputs.shape, labels.shape)

            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / total_train_samples

        model.eval()
        val_loss, total_correct_val, total_val_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _, cat_extras, cont_extras in valid_loader:
                images, labels = images.to(device), labels.to(device)
                cat_extras = [e.to(device) for e in cat_extras]
                cont_extras = [e.to(device) for e in cont_extras]

                outputs = model(images, cat_extras, cont_extras)
                val_loss += criterion(outputs, labels).item() * labels.size(0)
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)

        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / total_val_samples
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}, Time={epoch_time:.2f}s")

        log_epoch_to_csv(csv_file_path, epoch+1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        scheduler.step(avg_val_loss)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"New best accuracy: {best_accuracy:.4f}")
            # save vit part of model
            # torch.save(model.vit.state_dict(), os.path.join(checkpoint_dir, "vit_weights.pth"))

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_loss.pth"))
            print(f"New best loss: {best_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_accuracy, best_loss, tokenizers

# -------------------------
# Evaluation
# -------------------------
def evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session_name, categorical_keys, continuous_keys, tokenizers=None):
    
    ensure_folder(checkpoint_dir)

    best_trained_model = os.path.join(checkpoint_dir, "best_accuracy.pth")
    output_csv_path = os.path.join(checkpoint_dir, "test_predictions.csv")

    df = pd.read_csv(data_file)
    test_df = df[df['filename_index'].str.startswith('fungi_test')]

    # Tokenize categorical columns
    if categorical_keys and tokenizers is not None:
        test_df = apply_tokenizers(test_df, tokenizers)

    categorical_cardinalities = [len(tokenizers[k]) if k in tokenizers else 0 for k in categorical_keys]
    num_continuous = len(continuous_keys)

    test_dataset = FungiDataset(test_df, image_path, categorical_keys=categorical_keys, continuous_keys=continuous_keys, transform=get_transforms('valid'))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMultiModalFusion(
        num_classes=183,
        categorical_cardinalities=categorical_cardinalities,
        num_continuous=num_continuous,
        freeze_backbone=True
    )
    model.load_state_dict(torch.load(best_trained_model))
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for images, _, filenames, cat_extras, cont_extras in tqdm.tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            cat_extras = [e.to(device) for e in cat_extras]
            cont_extras = [e.to(device) for e in cont_extras]
            outputs = model(images, cat_extras, cont_extras).argmax(1).cpu().numpy()
            results.extend(zip(filenames, outputs))

    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([session_name])
        writer.writerows(results)
    print(f"Results saved to {output_csv_path}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    seed_torch(42)
    image_path = '/home/mmilo/FungiImages/'
    data_file = '/home/mmilo/MultimodalDataChallenge2025/metadata_v4.csv'

    # Define which keys are categorical and which are continuous
    categorical_keys = ["Habitat", "Substrate"]
    continuous_keys = ["Latitude", "Longitude"]

    results = []

    # Create tokenizers once for all runs
    df_all = pd.read_csv(data_file)
    tokenizers = create_categorical_tokenizers(df_all, categorical_keys)
    
    # 0. no metadata (baseline)
    # print("Training without any metadata")
    # session = "the_bear_no_metadata"
    # checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")
    # ensure_folder(checkpoint_dir)
    # acc, loss, used_tokenizers = train_fungi_network(
    #     data_file, image_path, checkpoint_dir, [], [], tokenizers=None
    # )
    # results.append((session, acc, loss))

    # 1. Each categorical key alone (no continuous)
    # for cat in categorical_keys:
    #     cats = [cat]
    #     conts = []
    #     print(f"Training with categorical: {cats}, continuous: {conts}")
    #     session = f"the_bear_{cat}"
    #     checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")

    #     acc, loss, used_tokenizers = train_fungi_network(
    #         data_file, image_path, checkpoint_dir, cats, conts, tokenizers=tokenizers if cats else None
    #     )
    #     results.append((session, acc, loss))
    #     evaluate_network_on_test_set(
    #         data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=used_tokenizers if cats else None
    #     )

    # # 2. Latitude and Longitude together (no categorical)
    # cats = []
    # conts = continuous_keys
    # print(f"Training with categorical: {cats}, continuous: {conts}")
    # session = f"the_bear_{'_'.join(conts)}"
    # checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")

    # acc, loss, used_tokenizers = train_fungi_network(
    #     data_file, image_path, checkpoint_dir, cats, conts, tokenizers=None
    # )
    # results.append((session, acc, loss))
    # evaluate_network_on_test_set(
    #     data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=None
    # )

    # 3. Optionally, all categorical keys together (no continuous)
    cats = categorical_keys
    conts = []
    print(f"Training with categorical: {cats}, continuous: {conts}")
    session = f"the_bear_focal"
    checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")

    acc, loss, used_tokenizers = train_fungi_network(
        data_file, image_path, checkpoint_dir, cats, conts, tokenizers=tokenizers if cats else None,
        n_epochs=100
    )
    results.append((session, acc, loss))
    
    evaluate_network_on_test_set(
        data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=used_tokenizers if cats else None
    )
    
    # train with all categorical and continuous keys
    # cats = categorical_keys
    # conts = continuous_keys
    # print(f"Training with categorical: {cats}, continuous: {conts}")
    # session = f"the_bear_{'_'.join(cats)}_{'_'.join(conts)}"
    # checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")
    # acc, loss, used_tokenizers = train_fungi_network(
    #     data_file, image_path, checkpoint_dir, cats, conts, tokenizers=tokenizers if cats else None
    # )
    # results.append((session, acc, loss))
    # evaluate_network_on_test_set(
    #     data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=used_tokenizers if cats else None
    # )

    # for session, acc, loss in results:
    #     print(f"Session: {session}\n Best Accuracy: {acc:.4f}, Best Loss: {loss:.4f}\n")
