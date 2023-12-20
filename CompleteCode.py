import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = pd.read_csv("./data/pima-indians-diabetes.csv")
X_np = dataset.iloc[:, 0:8].values
y_np = dataset.iloc[:, 8].values

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).reshape(-1, 1)


# define the model
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


model = PimaClassifier()
print(model)

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        # forward propagation
        X_batch = X[i : i + batch_size]
        y_pred = model(X_batch)
        y_batch = y[i : i + batch_size]
        loss = loss_fn(y_pred, y_batch)
        # back propagation
        optimizer.zero_grad()  # make previous gradients zero, else new gradient will be added to them.
        loss.backward()  # start computing gradient and back propagate
        optimizer.step()  # apply the computed gradients

# compute accuracy
y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print("%s => %d (expected %d)" % (X[i].tolist(), predictions[i], y[i]))
