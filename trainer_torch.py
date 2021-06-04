import torch
from torch import nn, optim
from NumpyTorch import transforms
import numpy as np
import scipy.io as scio

class MNIST(torch.utils.data.Dataset):
    def __init__(self, train: str="train"):
        f = scio.loadmat('digits.mat')
        self.train = (train == "train")
        if train == "train":
            data_x, data_y = f.get("X"), f.get("y")
        elif train == "valid":
            data_x, data_y = f.get("Xvalid"), f.get("yvalid")
        elif train == "test":
            data_x, data_y = f.get("Xtest"), f.get("ytest")
        """
        self.train_transform = transforms.Compose(
                transforms.RandomGamma(),
                transforms.Apply(
                        transforms.Selection([transforms.GaussianBlur(),
                                      transforms.LaplacianShapen(kernel_type="simple"),
                                      transforms.LaplacianShapen(kernel_type="full"),
                                      ], p = [0.5, 0.25, 0.25]),
                        p = 0.5
                ),
                transforms.Rotation(),
                transforms.Resize(),
                transforms.Translation()
                )
        """
        self.x, self.y = self.prepare(data_x, data_y)

    def prepare(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float32) / 255.
        x = x.reshape(-1, 1, 16, 16).transpose(0, 1, 3, 2)
        y = y.reshape(-1).astype(np.long) - 1
        return x, y

    def __getitem__(self, index:int):
        image, label = self.x[index, ...], self.y[index, ...]
        """
        # Data augmentation is activated when training
        if self.train is True:
            image = image.transpose(1, 2, 0)
            image = self.train_transform(image)
            image = image.transpose(2, 0, 1)
        """
        return image, label

    def __len__(self, ): return self.x.shape[0]


model = nn.Sequential(nn.Conv2d(1, 16, (3, 3), padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(16, 32, (3, 3), padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Conv2d(32, 64, (3, 3), padding=1),
                      nn.ReLU(),
                      nn.MaxPool2d(2, 2),
                      nn.Flatten(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Dropout(p=0.5),
                      nn.Linear(128, 10)
                      )

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0.)
TRAIN_DATASET = MNIST("train")
VALID_DATASET = MNIST("valid")
TEST_DATASET = MNIST("test")

TRAIN_LOADER = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=64, 
                                           shuffle=True, drop_last=True)
VALID_LOADER = torch.utils.data.DataLoader(VALID_DATASET, batch_size=1000, shuffle=False)
TEST_LOADER = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1000, shuffle=False)

for e in range(200):
    correct, total = 0, 0
    for batch_x, batch_y in TRAIN_LOADER:
        batch_x = batch_x.float()
        batch_y = batch_y.long()
        pred = model(batch_x)
        loss = criterion(pred, batch_y.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        _, pred_y = torch.max(pred, dim=1)
        correct += np.sum( (pred_y == batch_y).numpy() )
        total += batch_y.shape[0]
    train_correctness = correct / total * 100

    correct, total = 0, 0
    for batch_x, batch_y in VALID_LOADER:    
        pred = model(batch_x)

        _, pred_y = torch.max(pred, dim=1)
        correct += np.sum( (pred_y == batch_y).numpy() )
        total += batch_y.shape[0]
    valid_correctness = correct / total * 100
    print("Epoch:{:3d}, Training Accuracy:{:.2f}%, Validation Accuracy:{:.2f}%".format(
          e, train_correctness, valid_correctness
          ))

correct, total = 0, 0
for batch_x, batch_y in TEST_LOADER:    
    pred = model(batch_x)

    _, pred_y = torch.max(pred, dim=1)
    correct += np.sum( (pred_y == batch_y).numpy() )
    total += batch_y.shape[0]
test_correctness = correct / total * 100
print("Test Accuracy:{:.2f}%".format(test_correctness))