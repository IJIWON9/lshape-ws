import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from models.cnn1d import CNN1DClassifier

class ContourDataset(Dataset):
    def __init__(self, root_dir, input_length):
        self.samples = []
        self.labels = []
        self.input_length = input_length
        label_map = {"Bumper": 0, "SidePanel": 1}

        for filename in os.listdir(root_dir):
            if filename.endswith(".npy"):
                base = filename[:-4]
                npy_path = os.path.join(root_dir, base + ".npy")
                json_path = os.path.join(root_dir, base + ".json")
                if not os.path.exists(json_path):
                    continue

                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    label = meta.get("label", "Unknown")
                    if label not in label_map:
                        continue  # Unknown 무시

                data = np.load(npy_path)
                if len(data) != input_length:
                    continue

                self.samples.append(data)
                self.labels.append(label_map[label])

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx])  # shape: (1, 95)
        y = torch.tensor(self.labels[idx])
        return x, y


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = 95
    num_classes = 2
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    dataset = ContourDataset("./labeled_dataset", input_length)
    print(f"총 샘플 수: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = CNN1DClassifier(input_length, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()

        train_acc = correct / len(train_set)
        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct_val = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == y_val).sum().item()

        val_acc = correct_val / len(val_set)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/model_2class.pth")
    print("✅ 모델 저장 완료: weights/model_2class.pth")

if __name__ == "__main__":
    train()
