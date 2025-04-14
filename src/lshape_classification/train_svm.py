import os
import numpy as np
import json
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# 데이터 로드 함수
def load_dataset(path, input_length=95):
    X, y = [], []
    label_map = {"Bumper": 0, "SidePanel": 1}

    for fname in os.listdir(path):
        if fname.endswith('.npy'):
            npy_path = os.path.join(path, fname)
            json_path = npy_path.replace('.npy', '.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                meta = json.load(f)
                label = meta.get("label", "Unknown")
            if label not in label_map:
                continue
            vec = np.load(npy_path)
            if len(vec) != input_length:
                continue

            vec *= 10.0  # 🔥 inference와 동일하게 스케일 조정
            vec = (vec - np.mean(vec)) / (np.std(vec) + 1e-6)

            features = [
                np.max(vec),
                np.min(vec),
                np.mean(vec),
                np.std(vec),
                np.percentile(vec, 90),
                np.percentile(vec, 10),
                np.sum(np.abs(vec) > 0.6),
                np.nan_to_num(skew(vec)),        # NaN 방지
                np.nan_to_num(kurtosis(vec))     # NaN 방지
            ]
            print(f"[TRAIN DEBUG] Label: {label}, Feature: {features}")

            X.append(features)
            y.append(label_map[label])

    return np.array(X), np.array(y)

# 학습
X, y = load_dataset('./labeled_dataset')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='rbf', C=0.1, probability=True)
clf.fit(X_train, y_train)

# 평가
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Bumper", "SidePanel"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bumper", "SidePanel"])
disp.plot(cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()

# 모델 저장
os.makedirs("./weights", exist_ok=True)
joblib.dump(clf, "./weights/svm_model.pkl")
