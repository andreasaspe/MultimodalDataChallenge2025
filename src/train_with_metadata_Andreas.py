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
    """Create a mapping from category value to integer for each categorical column."""
    tokenizers = {}
    for key in categorical_keys:
        unique_vals = sorted(df[key].dropna().unique())
        tokenizers[key] = {val: idx for idx, val in enumerate(unique_vals)}
    return tokenizers

def apply_tokenizers(df, tokenizers):
    """Apply the categorical tokenizers to the DataFrame in-place."""
    for key, mapping in tokenizers.items():
        df[key] = df[key].map(lambda x: mapping.get(x, 0))  # Unknowns mapped to 0
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
        self.backbone.classifier = nn.Identity()
        # load weights from path
        # self.load_weights("/home/mmilo/FungiChallenge/results/EfficientNet_20250812_091422/best_accuracy.pth")

        

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

# -------------------------
# Training function
# -------------------------
def train_fungi_network(data_file, image_path, checkpoint_dir, categorical_keys, continuous_keys, tokenizers=None):
    ensure_folder(checkpoint_dir)
    csv_file_path = os.path.join(checkpoint_dir, 'train.csv')
    initialize_csv_logger(csv_file_path)

    df = pd.read_csv(data_file)
    train_df = df[df['filename_index'].str.startswith('fungi_train')]
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Tokenize categorical columns
    if categorical_keys:
        if tokenizers is None:
            tokenizers = create_categorical_tokenizers(df, categorical_keys)
        train_df = apply_tokenizers(train_df, tokenizers)
        val_df = apply_tokenizers(val_df, tokenizers)

    print('Training size', len(train_df))
    print('Validation size', len(val_df))

    categorical_cardinalities = [len(tokenizers[k]) if k in tokenizers else 0 for k in categorical_keys]
    num_continuous = len(continuous_keys)

    train_dataset = FungiDataset(train_df, image_path, categorical_keys=categorical_keys, continuous_keys=continuous_keys, transform=get_transforms('train'))
    valid_dataset = FungiDataset(val_df, image_path, categorical_keys=categorical_keys, continuous_keys=continuous_keys, transform=get_transforms('valid'))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalEfficientNet(
        num_classes=len(train_df['taxonID_index'].unique()),
        categorical_cardinalities=categorical_cardinalities,
        num_continuous=num_continuous,
        embed_dim=32,
        mlp_hidden=64
    ).to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0

    for epoch in range(100):
        model.train()
        train_loss, total_correct_train, total_train_samples = 0.0, 0, 0
        epoch_start_time = time.time()

        for images, labels, _, cat_extras, cont_extras in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            cat_extras = [e.to(device) for e in cat_extras]
            cont_extras = [e.to(device) for e in cont_extras]

            optimizer.zero_grad()
            outputs = model(images, cat_extras, cont_extras)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_correct_train += (outputs.argmax(1) == labels).sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = total_correct_train / total_train_samples
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss, total_correct_val, total_val_samples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, _, cat_extras, cont_extras in valid_loader:
                images, labels = images.to(device), labels.to(device)
                cat_extras = [e.to(device) for e in cat_extras]
                cont_extras = [e.to(device) for e in cont_extras]
                outputs = model(images, cat_extras, cont_extras)
                val_loss += criterion(outputs, labels).item()
                total_correct_val += (outputs.argmax(1) == labels).sum().item()
                total_val_samples += labels.size(0)

        val_accuracy = total_correct_val / total_val_samples
        avg_val_loss = val_loss / len(valid_loader)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}, "
              f"Time={epoch_time:.2f}s")

        log_epoch_to_csv(csv_file_path, epoch+1, epoch_time, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_accuracy.pth"))
            print(f"New best accuracy: {best_accuracy:.4f}")

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
    model = MultiModalEfficientNet(
        num_classes=len(df['taxonID_index'].unique()),
        categorical_cardinalities=categorical_cardinalities,
        num_continuous=num_continuous,
        embed_dim=32,
        mlp_hidden=64
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
    # Path to fungi images
    image_path = "/home/awias/data/Summerschool_2025/FungiImages"
    # Path to metadata file
    data_file = "/home/awias/data/Summerschool_2025/metadata.csv"

    # Define which keys are categorical and which are continuous
    categorical_keys = ["Habitat", "Substrate"]
    continuous_keys = ["Latitude", "Longitude"]

    results = []

    # Create tokenizers once for all runs
    df_all = pd.read_csv(data_file)
    tokenizers = create_categorical_tokenizers(df_all, categorical_keys)

    # 1. Each categorical key alone (no continuous)
    for cat in categorical_keys:
        cats = [cat]
        conts = []
        print(f"Training with categorical: {cats}, continuous: {conts}")
        session = f"EfficientNet_{cat}_none_{time.strftime('%Y%m%d_%H%M%S')}"
        checkpoint_dir = os.path.join(f"/home/awias/data/Summerschool_2025/results/{session}/")

        acc, loss, used_tokenizers = train_fungi_network(
            data_file, image_path, checkpoint_dir, cats, conts, tokenizers=tokenizers if cats else None
        )
        results.append((session, acc, loss))
        evaluate_network_on_test_set(
            data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=used_tokenizers if cats else None
        )

    # 2. Latitude and Longitude together (no categorical)
    cats = []
    conts = continuous_keys
    print(f"Training with categorical: {cats}, continuous: {conts}")
    session = f"EfficientNet_none_{'_'.join(conts)}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")

    acc, loss, used_tokenizers = train_fungi_network(
        data_file, image_path, checkpoint_dir, cats, conts, tokenizers=None
    )
    results.append((session, acc, loss))
    evaluate_network_on_test_set(
        data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=None
    )

    # 3. Optionally, all categorical keys together (no continuous)
    cats = categorical_keys
    conts = []
    print(f"Training with categorical: {cats}, continuous: {conts}")
    session = f"EfficientNet_{'_'.join(cats)}_none_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(f"/home/mmilo/FungiChallenge/results/{session}/")

    acc, loss, used_tokenizers = train_fungi_network(
        data_file, image_path, checkpoint_dir, cats, conts, tokenizers=tokenizers if cats else None
    )
    results.append((session, acc, loss))
    evaluate_network_on_test_set(
        data_file, image_path, checkpoint_dir, session, cats, conts, tokenizers=used_tokenizers if cats else None
    )

    for session, acc, loss in results:
        print(f"Session: {session}\n Best Accuracy: {acc:.4f}, Best Loss: {loss:.4f}\n")
