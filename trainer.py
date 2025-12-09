import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMG_WIDTH, IMG_HEIGHT = 380, 380  # EfficientNetB4 recommended size

# Training parameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.005
SEED = 42


# Paths
DATASET_DIR = Path("feathersv1-dataset")
MODEL_TYPE = "EfficientNetB4"
CLASSES_COUNT = "all"
PROJECT_NAME = f"{MODEL_TYPE}_{CLASSES_COUNT}"

DATA_CSV = DATASET_DIR / "data" / "feathers_data.csv"

df = pd.read_csv(DATA_CSV)

# Normalize species names (lowercase & replace spaces)
df['species'] = df['species'].str.lower().replace(" ", "_", regex=True)

# Construct full image paths
image_paths = (DATASET_DIR / "images" / df['order'].str.lower().replace(" ", "_", regex=True) /
               df['species'] / df['filename']).astype(str).tolist()
labels = df['species'].tolist()

enc = LabelEncoder()
labels_encoded = enc.fit_transform(labels)
label_counts = Counter(labels_encoded)

# This section of code is for only top 100 classes:

'''
top_100_classes = [class_idx for class_idx, _ in Counter(labels_encoded).most_common(100)]

# Keep only data points whose labels are in the top 100
top_100_indices = [i for i, label in enumerate(labels_encoded) if label in top_100_classes]

# Filter to keep only top 100 class data
image_paths = [image_paths[i] for i in top_100_indices]
labels = [labels[i] for i in top_100_indices]
labels_encoded = [labels_encoded[i] for i in top_100_indices]

# Re-encode labels to have consecutive indices from 0-99
enc = LabelEncoder()
labels_encoded = enc.fit_transform([labels[i] for i in range(len(labels))])
label_counts = Counter(labels_encoded)

'''

single_instance_indices = [i for i, label in enumerate(labels_encoded) if label_counts[label] == 1]
multi_instance_indices = [i for i, label in enumerate(labels_encoded) if label_counts[label] > 1]


single_image_paths = [image_paths[i] for i in single_instance_indices]
single_labels = [labels_encoded[i] for i in single_instance_indices]

# Get data for multi-instance classes
multi_image_paths = [image_paths[i] for i in multi_instance_indices]
multi_labels = [labels_encoded[i] for i in multi_instance_indices]


train_image_paths, test_image_paths, train_label_indices, test_label_indices = train_test_split(
    multi_image_paths, multi_labels, test_size=0.2, stratify=multi_labels, random_state=SEED
)

train_image_paths.extend(single_image_paths)
train_label_indices.extend(single_labels)
test_image_paths.extend(single_image_paths)  # Add same single-instance examples to test
test_label_indices.extend(single_labels)

# Convert lists to numpy arrays for compatibility
train_label_indices = np.array(train_label_indices)
test_label_indices = np.array(test_label_indices)



results_dir = Path("results") / PROJECT_NAME
results_dir.mkdir(parents=True, exist_ok=True)

def read_labels(file_path, label_type="species"):
    pos = 2 if label_type == "species" else 1
    with open(file_path, "r") as readfile:
        readfile.readline()  # Skip header
        return [line.strip().split(",")[pos] for line in readfile.readlines()]

# Get image paths
def csv_to_paths(dataset_dir, csv_file):
    with open(csv_file, "r") as readfile:
        readfile.readline()
        csv_data = readfile.readlines()

    image_paths, labels = [], []
    for line in csv_data:
        parts = line.strip().split(",")
        image_name, order, species = parts[0], parts[1].lower().replace(" ", "_"), parts[2].lower().replace(" ", "_")
        image_path = dataset_dir / "images" / order / species / image_name
        image_paths.append(image_path.as_posix())
        labels.append(species)

    return image_paths, labels


# Compute class weights
def calculate_class_weights(labels):
    df = pd.DataFrame({'species': labels})
    class_counts = df['species'].value_counts()
    n_samples = len(df)
    n_classes = len(class_counts)
    weights = {label: n_samples / (n_classes * count) for label, count in class_counts.items()}
    return weights

# Compute class weights based on training set only
species_weights = calculate_class_weights(train_label_indices)
class_weights = torch.tensor([species_weights.get(i, 1.0) for i in range(len(enc.classes_))], dtype=torch.float).to(device)


class FeatherDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_HEIGHT),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.CenterCrop(IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = FeatherDataset(train_image_paths, train_label_indices, transform=train_transform)
test_dataset = FeatherDataset(test_image_paths, test_label_indices, transform=test_transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor

for param in list(model.features.parameters())[-16:]:
    param.requires_grad = True
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(enc.classes_))



# Move model to device
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 32], gamma=0.5)

# Function to calculate Top-K Accuracy
def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        _, preds = output.topk(k, dim=1, largest=True, sorted=True)
        correct = preds.eq(target.view(-1, 1).expand_as(preds))
        return correct[:, :k].sum().item()
    
def validate(model, test_loader, criterion):
    model.eval()
    y_true, y_pred = [], []
    running_loss = 0.0
    correct, total = 0, 0
    top5_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Compute Top-5 accuracy
            top5_correct += top_k_accuracy(outputs, labels, k=5)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss = running_loss / len(test_loader)
    val_acc = 100 * correct / total
    val_top5_acc = 100 * top5_correct / total

    model.train()
    return val_loss, val_acc, val_top5_acc

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, epochs, scheduler):
    best_val_acc = 0.0  # Track best validation accuracy
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_top5_acc': []}

    model.train()
    
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate after every epoch
        val_loss, val_acc, val_top5_acc = validate(model, test_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top5_acc'].append(val_top5_acc)

        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-5 Acc: {val_top5_acc:.2f}%")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = results_dir / "best.pt"
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at {best_model_path} (Val Acc: {val_acc:.2f}%)")

        # The below code is for the 128 epochs training, I.E Experiment A

        # if epoch == 8:
        #     N = 5  # Number of last layers to unfreeze
        #     for param in list(model.features.parameters())[-N:]:  
        #         param.requires_grad = True

        # if epoch == 32:
        #     N=8
        #     for param in list(model.features.parameters())[-N:]:
        #         param.requires_grad = True

    return history

def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path / "loss_curve.png")
    plt.show()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.plot(epochs, history['val_top5_acc'], label='Validation Top-5 Accuracy', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(save_path / "accuracy_curve.png")
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    
    # Get only the labels present in y_true/y_pred
    unique_classes = sorted(set(y_true) | set(y_pred))  # Unique classes in test set
    target_names = [enc.classes_[i] for i in unique_classes]  # Get actual class names

    # Print classification report with correct classes
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=target_names))

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 15))  # Increase figure size for better readability
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save high-resolution image
    plt.show()

def load_model(model_path, model, optimizer, scheduler):
    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded from {model_path}")
    
    return model, optimizer, scheduler, checkpoint['epoch']



if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)  # Ensures compatibility on Windows

    # Train the model
    history = train_and_validate(model, train_loader, test_loader, criterion, optimizer, EPOCHS, scheduler)

    plot_training_curves(history, results_dir)

    # Evaluate the model
    best_model_path = results_dir / "best.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for evaluation")
    else:
        print("Best model not found, using the last trained model for evaluation")

    # Then continue with evaluation
    y_true, y_pred = evaluate_model(model, test_loader)
    y_true, y_pred = evaluate_model(model, test_loader)

    # Save confusion matri

    # Save the trained model
    model_save_path = results_dir / "efficientnet_b4_finetuned_top100.pt"
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'encoder_classes': enc.classes_
    }, model_save_path)

    print(f"Model saved to {model_save_path}")

