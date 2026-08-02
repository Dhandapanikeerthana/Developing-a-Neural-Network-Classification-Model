### Name: Keerthana D
### Register Number: 212224040155

# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import the required Python libraries such as PyTorch, Pandas, NumPy, Matplotlib, and Scikit-learn.

### STEP 2: 

Load the customer dataset, remove unnecessary columns, handle missing values, and encode categorical features.

### STEP 3: 

Split the dataset into training and testing sets and normalize the input features using StandardScaler.

### STEP 4: 

Convert the data into PyTorch tensors and create DataLoaders for batch training.

### STEP 5: 

Define the neural network model, initialize the loss function (CrossEntropyLoss), and optimizer (Adam).

### STEP 6: 

Train the neural network, evaluate it using the test dataset, and generate the confusion matrix and classification report.



## PROGRAM


```python
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

df = pd.read_csv(r"C:\Users\Jayani N\Downloads\PyTorch-Ultimate-2023---From-Basics-to-Cutting-Edge\010_DeepLearningIntro\customers.csv")

df.drop(columns=["ID"], inplace=True)

# ---------------------------------------------------
# Handle Missing Values
# ---------------------------------------------------

df["Work_Experience"].fillna(0, inplace=True)
df["Family_Size"].fillna(df["Family_Size"].median(), inplace=True)

# ---------------------------------------------------
# Encode Target
# ---------------------------------------------------

label_encoder = LabelEncoder()
df["Segmentation"] = label_encoder.fit_transform(df["Segmentation"])

# ---------------------------------------------------
# One-Hot Encode Categorical Features
# ---------------------------------------------------

x = pd.get_dummies(
    df.drop(columns=["Segmentation"]),
    columns=[
        "Gender",
        "Ever_Married",
        "Graduated",
        "Profession",
        "Spending_Score",
        "Var_1"
    ]
)

y = df["Segmentation"].values

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

xt, xst, yt, yst = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------

scaler = StandardScaler()

xt = scaler.fit_transform(xt)
xst = scaler.transform(xst)

# ---------------------------------------------------
# Convert to Torch
# ---------------------------------------------------

xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)

yt = torch.LongTensor(yt)
yst = torch.LongTensor(yst)

# ---------------------------------------------------
# DataLoader
# ---------------------------------------------------

train_dataset = TensorDataset(xt, yt)
test_dataset = TensorDataset(xst, yst)

train_loader = DataLoader(train_dataset,
                          batch_size=32,
                          shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=32,
                         shuffle=False)

# ---------------------------------------------------
# Neural Network
# ---------------------------------------------------

class Classifier(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.l1 = nn.Linear(input_size, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.l4 = nn.Linear(16, 4)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = self.dropout(x)

        x = F.relu(self.l2(x))
        x = self.dropout(x)

        x = F.relu(self.l3(x))

        x = self.l4(x)

        return x

# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = Classifier(input_size=xt.shape[1])

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ---------------------------------------------------
# Training
# ---------------------------------------------------

epochs = 300

loss_history = []

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        outputs = model(xb)

        loss = criterion(outputs, yb)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    loss_history.append(epoch_loss)

    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{epochs}  Loss : {epoch_loss:.4f}")

# ---------------------------------------------------
# Loss Graph
# ---------------------------------------------------

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# ---------------------------------------------------
# Testing
# ---------------------------------------------------

model.eval()

predictions = []
actual = []

with torch.no_grad():

    for xb, yb in test_loader:

        outputs = model(xb)

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.numpy())

        actual.extend(yb.numpy())

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

accuracy = accuracy_score(actual, predictions)

print("\nAccuracy :", accuracy)

print("\nConfusion Matrix\n")

print(confusion_matrix(actual, predictions))

print("\nClassification Report\n")

print(classification_report(
    actual,
    predictions,
    target_names=label_encoder.classes_
))

# ---------------------------------------------------
# Sample Predictions
# ---------------------------------------------------

print("\nFirst 20 Predictions\n")

for i in range(20):

    print(
        f"Actual : {label_encoder.inverse_transform([actual[i]])[0]}"
        f"   Predicted : {label_encoder.inverse_transform([predictions[i]])[0]}"
    )
# %%


```

### Dataset Information
<img width="740" height="961" alt="image" src="https://github.com/user-attachments/assets/afd19214-3972-4a04-a656-9e13d8656d5e" />


### OUTPUT

## Confusion Matrix

<img width="339" height="154" alt="image" src="https://github.com/user-attachments/assets/075e65cf-4341-467b-a1e6-5f21c9cf9b08" />


## Classification Report
<img width="570" height="316" alt="image" src="https://github.com/user-attachments/assets/4cc0f3a3-ef7f-4a20-9c8e-074e36b5d5f7" />


### New Sample Data Prediction
<img width="664" height="142" alt="image" src="https://github.com/user-attachments/assets/6caa58d6-71ec-4f7c-a4c1-4ca60cd3581e" />
<img width="781" height="877" alt="image" src="https://github.com/user-attachments/assets/b359294a-ec13-4b6a-be8e-1f53a8eabcf4" />


## RESULT
A neural network classification model was successfully developed using PyTorch to classify customers into four segments (A, B, C, and D).
