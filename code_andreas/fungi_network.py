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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision import models
from sklearn.model_selection import train_test_split
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
from tqdm import tqdm
import numpy as np
from PIL import Image
import time
import csv
from collections import Counter
import wandb
import ast

class tokenize_attributes():
    """
    Tokenize attributes based on their type.
    """
    def __init__(self, cameraModelSTxt, CameraMakerTxt):
        self.habitat_types = ['Mixed woodland (with coniferous and deciduous trees)', 'Unmanaged deciduous woodland',
                              'Forest bog', 'coniferous woodland/plantation', 'Deciduous woodland', 'natural grassland', 'lawn',
                              'Unmanaged coniferous woodland', 'garden', 'wooded meadow, grazing forest', 'dune', 'Willow scrubland', 'heath',
                              'Acidic oak woodland', 'roadside', 'Thorny scrubland', 'park/churchyard', 'Bog woodland', 'hedgerow', 'gravel or clay pit',
                              'salt meadow', 'bog', 'meadow', 'improved grassland', 'other habitat', 'roof', 'fallow field', 'ditch', 'fertilized field in rotation']
        
        self.substrate_types = ['soil', 'leaf or needle litter', 'wood chips or mulch', 'dead wood (including bark)', 'bark',
                                'wood', 'bark of living trees', 'mosses', 'wood and roots of living trees', 'stems of herbs, grass etc',
                                'peat mosses','dead stems of herbs, grass etc', 'fungi', 'other substrate', 'living stems of herbs, grass etc',
                                'living leaves', 'fire spot', 'faeces', 'cones', 'fruits']
        #load the txt files with camera models and camera makers, the text file is already on the form of a list of strings
        with open(cameraModelSTxt, "r", encoding="utf-8") as f:
            camera_models_types = f.read()
        self.camera_models_types = ast.literal_eval(camera_models_types)
        with open(CameraMakerTxt, "r", encoding="utf-8") as f:
            camera_makers_types = f.read()
        self.camera_makers_types = ast.literal_eval(camera_makers_types)
    
    
        self.habitat_types2idx = {habitat: idx for idx, habitat in enumerate(self.habitat_types)}
        self.substrate_types2idx = {substrate: idx for idx, substrate in enumerate(self.substrate_types)}
        self.camera_models2idx = {model: idx for idx, model in enumerate(self.camera_models_types)}
        self.camera_makers2idx = {maker: idx for idx, maker in enumerate(self.camera_makers_types)}
        
        self.num_habitats = len(self.habitat_types) + 1  # +1 for 'missing habitat'
        self.num_substrates = len(self.substrate_types) + 1  # +1 for 'missing substrate'
        self.num_months = 12+1
        self.num_days = 31+1 # 0-23, +1 for 'missing hour'
        self.num_camera_models = len(self.camera_models_types) + 1  # +1 for 'missing camera model'
        self.num_camera_makers = len(self.camera_makers_types) + 1  # +1 for 'missing camera maker'
        
        
    
    def tokenize(self, attribute, attribute_type):
        if attribute_type == 'Habitat':
            if attribute not in self.habitat_types:
                return len(self.habitat_types)  # Return index for 'missing habitat'
            else:
                return self.habitat_types2idx[attribute]
        elif attribute_type == 'Substrate':
            if attribute not in self.substrate_types:
                return len(self.substrate_types)
            else:
                return self.substrate_types2idx[attribute]
        elif attribute_type == 'DateTimeOriginal':
            try:
                # EXIF-like "YYYY:MM:DD HH:MM:SS"
                yymmdd, hhmmss = attribute.split(' ')
                month = int(yymmdd.split(':')[1])   # 1..12
                day  = int(yymmdd.split(':')[2])   # 0..23

                # sanitize ranges
                if not (1 <= month <= 12):
                    month_idx = self.num_months - 1   # missing month
                else:
                    month_idx = month - 1             # 0..11

                if not (1 <= day <= 31):
                    day_idx = self.num_days - 1     # missing hour
                else:
                    day_idx = day                   # 0..23

                return month_idx, day_idx
            except Exception:
                # fallback: both missing
                return self.num_months - 1, self.num_days - 1
        elif attribute_type == 'camera_model':
            if attribute not in self.camera_models_types:
                return len(self.camera_models_types)
            else:
                return self.camera_models2idx[attribute]
        elif attribute_type == 'camera_maker':
            if attribute not in self.camera_makers_types:
                return len(self.camera_makers_types)
            else:
                return self.camera_makers2idx[attribute]
            

def ensure_folder(folder):
    """
    Ensure a folder exists; if not, create it.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist. Creating...")
        os.makedirs(folder)

def seed_torch(seed=777):
    """
    Set seed for reproducibility.
    """
    random.seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def initialize_csv_logger(file_path):
    """Initialize the CSV file with header."""
    header = ["epoch", "time", "val_loss", "val_accuracy", "train_loss", "train_accuracy"]
    with open(file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

def log_epoch_to_csv(file_path, epoch, epoch_time, train_loss, train_accuracy, val_loss, val_accuracy):
    """Log epoch summary to the CSV file."""
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([epoch, epoch_time, val_loss, val_accuracy, train_loss, train_accuracy])

def get_transforms(data):
    """
    Return augmentation transforms for the specified mode ('train' or 'valid').
    """
    width, height = 224, 224
    if data == 'train':
        return Compose([
            RandomResizedCrop(size = (width, height), scale=(0.8, 1.0)),
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
        raise ValueError("Unknown data mode requested (only 'train' or 'valid' allowed).")

class FungiDataset(Dataset):
    def __init__(self, df, path, cameraModelSTxt, CameraMakerTxt, transform=None, multi_modal=False):
        self.df = df
        self.transform = transform
        self.path = path
        self.multi_modal = multi_modal
        self.tokenizer = tokenize_attributes(cameraModelSTxt, CameraMakerTxt)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['filename_index'].values[idx]
        # Get label if it exists; otherwise return None
        label = self.df['taxonID_index'].values[idx]  # Get label
        if pd.isnull(label):
            label = -1  # Handle missing labels for the test dataset
        else:
            label = int(label)
            
        if self.multi_modal:
            habitat = self.df['Habitat'].values[idx]
            if pd.isnull(habitat):
                habitat = -1
            else:
                habitat = str(habitat)
            habitat = self.tokenizer.tokenize(habitat, 'Habitat')
                
            latitude = self.df['Latitude'].values[idx]
            if pd.isnull(latitude):
                latitude = -1
            else:
                latitude = float(latitude)
            longitude = self.df['Longitude'].values[idx]
            if pd.isnull(longitude):
                longitude = -1
            else:
                longitude = float(longitude)
            substrate = self.df['Substrate'].values[idx]
            if pd.isnull(substrate):
                substrate = -1
            else:
                substrate = str(substrate)
            substrate = self.tokenizer.tokenize(substrate, 'Substrate')
            eventDate = self.df['DateTimeOriginal'].values[idx]
            if pd.isnull(eventDate):
                month, hour = self.tokenizer.num_months - 1, self.tokenizer.num_hours - 1
            else:
                eventDate = str(eventDate)
                month, hour = self.tokenizer.tokenize(eventDate, 'DateTimeOriginal')
                
            cameraModel = self.df['camera_model'].values[idx]
            if pd.isnull(cameraModel):
                cameraModel = self.tokenizer.num_camera_models
            else:
                cameraModel = str(cameraModel)
            cameraModel = self.tokenizer.tokenize(cameraModel, 'camera_model')
            cameraMaker = self.df['camera_maker'].values[idx]
            if pd.isnull(cameraMaker):
                cameraMaker = self.tokenizer.num_camera_makers
            else:
                cameraMaker = str(cameraMaker)
            cameraMaker = self.tokenizer.tokenize(cameraMaker, 'camera_maker')
            
        with Image.open(os.path.join(self.path, file_path)) as img:
            # Convert to RGB mode (handles grayscale images as well)
            image = img.convert('RGB')
        image = np.array(image)

        # Apply transformations if available
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if self.multi_modal:
            return image, label, file_path, habitat, substrate, month, hour, cameraModel, cameraMaker, latitude, longitude
        else:
            return image, label, file_path

class AttributeEmbedder(nn.Module):
    """
    Embeds additional attributes (habitat, substrate, eventDate) into a fixed-size vector.
    """
    def __init__(self, num_habitats, num_substrates, num_months, num_hours,
                 num_camera_models, num_camera_makers, num_embedding_dims=64):
        super(AttributeEmbedder, self).__init__()
        self.habitat_embedding = nn.Embedding(num_habitats, num_embedding_dims)
        self.substrate_embedding = nn.Embedding(num_substrates, num_embedding_dims)
        self.month_embedding = nn.Embedding(num_months, num_embedding_dims)
        self.hour_embedding = nn.Embedding(num_hours, num_embedding_dims)  # 24 hours in a day
        self.camera_model_embedding = nn.Embedding(num_camera_models, num_embedding_dims)
        self.camera_maker_embedding = nn.Embedding(num_camera_makers, num_embedding_dims)
        self.geo_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, num_embedding_dims)
        )

    def forward(self, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude):
        h = self.habitat_embedding(habitat)         # (B, E)
        s = self.substrate_embedding(substrate)     # (B, E)
        m = self.month_embedding(month)   # (B, E)
        hr = self.hour_embedding(hour)  # (B, E)
        cmod = self.camera_model_embedding(camera_model)  # (B, E)
        cmak = self.camera_maker_embedding(camera_maker)  # (B, E)
        g = self.geo_mlp(torch.stack([latitude, longitude], dim=1))  # (B, E)
        return torch.cat([h, s, m, hr, cmod, cmak, g], dim=1)       # (B, 4E)

class MultiModalEffNet(nn.Module):
    def __init__(self, num_classes, attr_embedder, attr_dim=64*7):  # 4E from above
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # EfficientNet-B0 final feature size is 1280 before the classifier:
        in_features = self.backbone.classifier[1].in_features
        # strip the final Linear
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Identity()   # will emit the 1280-d pooled features
        )
        self.attr = attr_embedder
        self.head = nn.Sequential(
            nn.Linear(in_features + attr_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude):
        img_feat = self.backbone(images)  # (B, 1280)
        attr_feat = self.attr(habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude)  # (B, 4E)
        x = torch.cat([img_feat, attr_feat], dim=1)
        return self.head(x)
    


def train_fungi_network(data_file, image_path, checkpoint_dir, multi_modal=False, wandb_bool=False):
    """
    Train the network and save the best models based on validation accuracy and loss.
    Incorporates early stopping with a patience of 10 epochs.
    """
    # Ensure checkpoint directory exists
    CameraModelSTxt = 'C:/Users/bmsha/sc2025/metadata_1/camera_models.txt'
    CameraMakerTxt = 'C:/Users/bmsha/sc2025/metadata_1/camera_makers.txt'
    
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Network Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = train_df['taxonID_index'].nunique()
    if multi_modal:
        csv_embedder = AttributeEmbedder(
            num_habitats=train_dataset.tokenizer.num_habitats,
            num_substrates=train_dataset.tokenizer.num_substrates,
            num_months=train_dataset.tokenizer.num_months,
            num_hours=train_dataset.tokenizer.num_hours,
            num_camera_models=train_dataset.tokenizer.num_camera_models,
            num_camera_makers=train_dataset.tokenizer.num_camera_makers,
            num_embedding_dims=64
        )
        model = MultiModalEffNet(num_classes=num_classes,
                                attr_embedder=csv_embedder,
                                attr_dim=64*7).to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, eps=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Early stopping setup
    patience = 10
    patience_counter = 0
    best_loss = np.inf
    best_accuracy = 0.0
    if wandb_bool:
        wandb.init(project="sc2025", entity="Bjonze", name="EfficientNet_B0_Weights")

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
                images, labels, _, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude = batch
                images = images.to(device); labels = labels.to(device).long()
                habitat = habitat.to(device).long(); substrate = substrate.to(device).long()
                month = month.to(device).long()
                hour = hour.to(device).long()
                camera_model = camera_model.to(device).long(); camera_maker = camera_maker.to(device).long()
                latitude = latitude.to(device).float(); longitude = longitude.to(device).float()

                optimizer.zero_grad()
                outputs = model(images, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude)
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
                    images, labels, _, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude = batch
                    images = images.to(device); labels = labels.to(device).long()
                    habitat = habitat.to(device).long(); substrate = substrate.to(device).long()
                    month = month.to(device).long(); hour = hour.to(device).long()
                    camera_model = camera_model.to(device).long(); camera_maker = camera_maker.to(device).long()
                    latitude = latitude.to(device).float(); longitude = longitude.to(device).float()

                    optimizer.zero_grad()
                    outputs = model(images, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude)
                    
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
    
    if multi_modal:
        csv_embedder = AttributeEmbedder(
            num_habitats=test_dataset.tokenizer.num_habitats,
            num_substrates=test_dataset.tokenizer.num_substrates,
            num_months=test_dataset.tokenizer.num_months,
            num_hours=test_dataset.tokenizer.num_hours,
            num_camera_models=test_dataset.tokenizer.num_camera_models,
            num_camera_makers=test_dataset.tokenizer.num_camera_makers,
            num_embedding_dims=64
        )
        model = MultiModalEffNet(num_classes=num_classes,
                                attr_embedder=csv_embedder,
                                attr_dim=64*7).to(device)
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        model = model.to(device)
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
                images, _, filenames, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude = batch
                images = images.to(device)
                habitat = habitat.to(device).long(); substrate = substrate.to(device).long()
                month = month.to(device).long(); hour = hour.to(device).long()
                camera_model = camera_model.to(device).long(); camera_maker = camera_maker.to(device).long()
                latitude = latitude.to(device).float(); longitude = longitude.to(device).float()

                outputs = model(images, habitat, substrate, month, hour, camera_model, camera_maker, latitude, longitude)
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
    image_path = 'C:/Users/bmsha/sc2025/FungiImages'
    # Path to metadata file
    data_file = str('C:/Users/bmsha/sc2025/metadata_1/metadata_fused.csv')

    # Session name: Change session name for every experiment! 
    # Session name will be saved as the first line of the prediction file
    session = "EfficientNet_cameraInfo"
    wandb_bool = True
    # Folder for results of this experiment based on session name:
    checkpoint_dir = os.path.join(f"C:/Users/bmsha/sc2025/checkpoints/{session}/")

    train_fungi_network(data_file, image_path, checkpoint_dir, multi_modal=True, wandb_bool=wandb_bool)
    wandb.finish() if wandb_bool else None
    evaluate_network_on_test_set(data_file, image_path, checkpoint_dir, session, multi_modal=True)