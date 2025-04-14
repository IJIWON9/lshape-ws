import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from models.cnn1d import CNN1DClassifier

class ContourDataset(Dataset):
    def __init__(self, root, input_length):
        self.samples = []
        label_map = {"Bumper": 0, "SidePanel": 1}
        for fname in os.listdir(root):
            if fname.endswith(".npy"):
                npy_path = os.path.join(root, fname)
                json_path = npy_path.replace(".npy", ".json")
                if not os.path.exists(json_path):
                    continue
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    label = meta.get("label", "Unknown")
                if label not in label_map:
                    continue
                data = np.load(npy_path)
                if len(data) != input_length:
                    continue
                # 🔥 진폭 크게 만들기
                data *= 10.0
                # data = data / (np.max(np.abs(data)) + 1e-6)
                self.samples.append((data.astype(np.float32), label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


def train():
    input_length = 95
    batch_size = 32
    num_epochs = 30
    lr = 0.001

    dataset = ContourDataset("./labeled_dataset", input_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = CNN1DClassifier(input_length=input_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

    # Save model
    os.makedirs("./weights", exist_ok=True)
    torch.save(model.state_dict(), "./weights/cnn1d_model.pth")

if __name__ == "__main__":
    train()
