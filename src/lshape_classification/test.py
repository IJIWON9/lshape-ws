import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from models.cnn1d import CNN1DClassifier
from datasets.contour_dataset import Contour1DDataset

if __name__ == "__main__":
    data_dir = "./labeled_dataset"
    input_length = 95
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Contour1DDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32)

    model = CNN1DClassifier(input_length, num_classes).to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            pred = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(y.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Bumper", "SidePanel", "Unknown"]))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
