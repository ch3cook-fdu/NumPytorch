import numpy as np
from NumpyTorch import nn, optim, utils, transforms
import scipy.io as scio
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Trainer details for CNN on MNIST.')
parser.add_argument('--config', type=str, default="conv3-mlp2", help='model definition in: config/*.txt')
parser.add_argument('--seed', type=int, default=123, help='random seed to garantee reproductivity')
# trainer information
parser.add_argument('--num_epoch', type=int, default=200, help='training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-5)
args = parser.parse_args()

# fix random seed
np.random.seed(args.seed)
test_interval = 25
filename = "checkpoints/classifier-{}.pt".format(args.config)

class MNIST(utils.Dataset):
    def __init__(self, train: str="train"):
        f = scio.loadmat('digits.mat')
        self.train = (train == "train")
        if train == "train":
            data_x, data_y = f.get("X"), f.get("y")
        elif train == "valid":
            data_x, data_y = f.get("Xvalid"), f.get("yvalid")
        elif train == "test":
            data_x, data_y = f.get("Xtest"), f.get("ytest")

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
        self.x, self.y = self.prepare(data_x, data_y)

    def prepare(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float32) / 255.
        x = x.reshape(-1, 1, 16, 16).transpose(0, 1, 3, 2)
        y = y.reshape(-1).astype(np.long) - 1
        return x, y

    def __getitem__(self, index:int):
        image, label = self.x[index, ...], self.y[index, ...]
        # Data augmentation is activated when training
        if self.train is True:
            image = image.transpose(1, 2, 0)
            image = self.train_transform(image)
            image = image.transpose(2, 0, 1)
        return image, np.array(label)

    def __len__(self, ): return self.x.shape[0]

# Defining dataset and dataloader
TRAIN_DATASET = MNIST("train")
VALID_DATASET = MNIST("valid")
TEST_DATASET = MNIST("test")
#   drop last during training, since the last batch is much smaller than previous
# batches, which may affect the convergence and perfomance of the model
TRAIN_LOADER = utils.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, 
                                shuffle=True, drop_last=True)
VALID_LOADER = utils.DataLoader(VALID_DATASET, batch_size=1000, shuffle=False)
TEST_LOADER = utils.DataLoader(TEST_DATASET, batch_size=1000, shuffle=False)

# evaluate the checkpoint
def evaluate(model, dataloader):
    correct, total = 0, 0
    for batch_x, batch_y in dataloader:    
        pred = model(batch_x)
        pred_y = np.argmax(pred, axis=1)
        correct += np.sum(pred_y == batch_y)
        total += batch_y.shape[0]
    return correct / total * 100

# Define model from the config file
with open("config/{}.txt".format(args.config), "r") as file:
    classifier = eval(file.read())

print("=============== Network Architecture: ===============\n{}".format(classifier))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=500, eta_min=0., verbose=False)

for e in range(args.num_epoch):
    classifier.train()
    correct, total = 0, 0
    for batch_x, batch_y in tqdm(TRAIN_LOADER, ncols=50):
        pred = classifier(batch_x)
        loss = criterion(pred, batch_y)
    
        classifier.backward(criterion)
        optimizer.step()

        pred_y = np.argmax(pred, axis=1)
        correct += np.sum(pred_y == batch_y)
        total += batch_y.shape[0]
    train_correctness = correct / total * 100
    scheduler.step()

    # evalutate to monitor the training schedule
    classifier.eval()
    valid_correctness = evaluate(classifier, VALID_LOADER)
    print("Epoch:{:3d}, Training Accuracy:{:.2f}%, Validation Accuracy:{:.2f}%".format(
          e, train_correctness, valid_correctness
          ))

# save checkpoint
print("saving model....")
state_dict = classifier.get_state_dict()
utils.save(state_dict, filename)
# Test the trained model
classifier.eval()
s = time.time()
test_correctness = evaluate(classifier, TEST_LOADER)
print("Test Accuracy: {:.2f}%, inference time: {:.4f} seconds".format(test_correctness,
      time.time() - s))

# %% Fine tune the model's last linear layer
print("============================ Fine Tuning ============================")
print("loading pretrained weights....")

state_dict = utils.load(filename)
classifier.load_state_dict(state_dict)

last_layer = next(reversed(classifier.ModuleDict))
classifier.ModuleDict[last_layer] = nn.Linear(128, 10)
optimizer = optim.SGD(classifier[last_layer], lr=1e-2, momentum=0.9, weight_decay=1e-4)

for e in range(20):
    classifier.train()
    correct, total = 0, 0
    for batch_x, batch_y in TRAIN_LOADER:
        pred = classifier(batch_x)
        loss = criterion(pred, batch_y)

        classifier.backward(criterion)
        optimizer.step()

        pred_y = np.argmax(pred, axis=1)
        correct += np.sum(pred_y == batch_y)
        total += batch_y.shape[0]
    train_correctness = correct / total * 100
    # evalutate to monitor the training schedule
    classifier.eval()
    valid_correctness = evaluate(classifier, VALID_LOADER)
    print("Epoch:{:3d}, Training Accuracy:{:.2f}%, Validation Accuracy:{:.2f}%".format(
          e, train_correctness, valid_correctness
          ))

# %% Test the fine tuned trained model
classifier.eval()
s = time.time()
test_correctness = evaluate(classifier, TEST_LOADER)
print("Test Accuracy: {:.2f}%, inference time: {:.4f} seconds".format(test_correctness,
      time.time() - s))