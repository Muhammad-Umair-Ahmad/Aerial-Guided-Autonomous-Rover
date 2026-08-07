import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── 1. Define the CNN Model ──
class RoverCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(RoverCNN, self).__init__()
        # Input: 3 x 128 x 128
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # 16 x 64 x 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 32 x 32
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32 x 8 x 8
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64 x 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ── 2. Define the Dataset ──
class RoverDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        
        # Define command mapping to integers
        self.cmd_to_idx = {
            "forward": 0,
            "reverse": 1,
            "left": 2,
            "right": 3,
            "stop": 4
        }
        
        # Load CSV
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        img_name, cmd = row[0], row[1]
                        if cmd in self.cmd_to_idx and os.path.exists(os.path.join(img_dir, img_name)):
                            self.data.append((img_name, self.cmd_to_idx[cmd]))
                            
        print(f"Loaded {len(self.data)} training samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ── 3. Training Loop ──
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Augmentation & Preprocessing
    # We augment the data slightly to prevent overfitting on 3-4 manual runs
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_dir = "dataset"
    csv_file = os.path.join(dataset_dir, "labels.csv")
    img_dir = os.path.join(dataset_dir, "images")
    
    dataset = RoverDataset(csv_file, img_dir, transform=transform)
    if len(dataset) == 0:
        print("No data to train on! Please collect data first using the dashboard.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = RoverCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
        
    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/behavioral_clone.pth")
    print("Training complete! Model saved to models/behavioral_clone.pth")

if __name__ == "__main__":
    train()
