import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from models.cnn1d import CNN1DClassifier
from datasets.contour_dataset import Contour1DDataset

if __name__ == '__main__':
    data_dir = "./labeled_dataset"
    input_length = 95
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Contour1DDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CNN1DClassifier(input_length, num_classes).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        out = model(x)
        pred = out.argmax(dim=1).item()
        true = y.item()

        if pred != true:
            print(f"❌ Misclassified Sample #{idx}: Pred={pred}, GT={true}")
            plt.figure(figsize=(8, 2))
            plt.ylim(-0.8, 0.8)
            plt.plot(x.cpu().numpy().squeeze(), marker='o')
            plt.title(f"Misclassified: Pred={pred}, GT={true}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
