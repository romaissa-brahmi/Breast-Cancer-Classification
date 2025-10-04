# --------------------
# Import the libraries
# --------------------
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch.optim as optim


# -------------
# Load the data
# -------------
breast_cancer_dataset = fetch_ucirepo(id=17)
X = breast_cancer_dataset.data.features
y = breast_cancer_dataset.data.targets

dataset = pd.concat([X, y], axis=1)
data_B = dataset[dataset['Diagnosis'] == 'B']
data_M = dataset[dataset['Diagnosis'] == 'M']

sample_number = min(len(data_B), len(data_M))
data_B = data_B.sample(n=sample_number, random_state=12)
data_M = data_M.sample(n=sample_number, random_state=12)
balanced_dataset = pd.concat([data_B, data_M])


# -------------
# Preprocessing
# -------------
X = balanced_dataset.drop(['Diagnosis'], axis=1)
y = balanced_dataset['Diagnosis']
y = y.map({'B':0, 'M':1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)

standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)

training_dataset = TensorDataset(X_train, y_train)
testing_dataset = TensorDataset(X_test, y_test)

batch_size = 12
train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)


# ------------------------
# Build the Neural Network
# ------------------------
class Classification(nn.Module):
    def __init__(self, INPUT_NB, HIDDEN_NB, OUTPUT_NB):
        super().__init__()
        self.layer1 = nn.Linear(INPUT_NB, HIDDEN_NB)
        self.layer2 = nn.Linear(HIDDEN_NB, OUTPUT_NB)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


INPUT_NB = X.shape[1]
HIDDEN_NB = 32
OUTPUT_NB = len(y.unique())
learning_rate = 0.001

model = Classification(INPUT_NB, HIDDEN_NB, OUTPUT_NB)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# --------------------
# Training and testing
# --------------------
epochs = 30
train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0

    for X_batch, y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train_correct += (predicted == y_batch).sum().item()

    average_train_accuracy = total_train_correct / len(training_dataset)
    average_train_loss = total_train_loss / len(train_loader)

    train_accuracy.append(average_train_accuracy)
    train_loss.append(average_train_loss)

    model.eval()
    total_test_loss = 0.0
    total_test_correct = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_test_correct += (predicted == y_batch).sum().item()

    average_test_accuracy = total_test_correct / len(testing_dataset)
    average_test_loss = total_test_loss / len(test_loader)

    test_accuracy.append(average_test_accuracy)
    test_loss.append(average_test_loss)


# ------------------------
# Get the confusion matrix
# ------------------------
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(y_batch.tolist())


# ----------------
# Plot the results
# ----------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].plot(range(1, epochs + 1), train_loss, label='Training Loss')
axs[0].plot(range(1, epochs + 1), test_loss, label='Test Loss', linestyle='--')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Test Loss Curve')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy')
axs[1].plot(range(1, epochs + 1), test_accuracy, label='Test Accuracy', linestyle='--')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Training and Test Accuracy Curve')
axs[1].legend()
axs[1].grid(True)


cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap="Greens", ax=axs[2], colorbar=False)
axs[2].set_title("Confusion Matrix for the last Epoch")

plt.tight_layout()
plt.show()

