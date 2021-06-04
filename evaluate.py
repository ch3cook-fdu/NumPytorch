import numpy as np
from NumpyTorch import utils, nn
import scipy.io as scio
import heapq
import matplotlib.pyplot as plt
import time
import argparse

# fix random seed
parser = argparse.ArgumentParser(description='Trainer details for CNN on MNIST.')
parser.add_argument('--config', type=str, default="conv3-mlp2", help='model definition in: config/*.txt')
parser.add_argument('--dir', type=str, default="checkpoints/with augmentation", help='checkpoint directory')
args = parser.parse_args()


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
        self.x, self.y = self.prepare(data_x, data_y)

    def prepare(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float32) / 255.
        x = x.reshape(-1, 1, 16, 16).transpose(0, 1, 3, 2)
        y = y.reshape(-1).astype(np.long) - 1
        return x, y

    def __getitem__(self, index:int):
        image, label = self.x[index, ...], self.y[index, ...]
        return image, label

    def __len__(self, ): return self.x.shape[0]


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

VALID_DATASET = MNIST("valid")
TEST_DATASET = MNIST("test")

VALID_LOADER = utils.DataLoader(VALID_DATASET, batch_size=5000, shuffle=False)
TEST_LOADER = utils.DataLoader(TEST_DATASET, batch_size=1000, shuffle=False)

state_dict = utils.load("{}/classifier-{}.pt".format(args.dir, args.config))
classifier.load_state_dict(state_dict)

classifier.eval()
s = time.time()
test_correctness = evaluate(classifier, TEST_LOADER)
print("Test Accuracy: {:.2f}%, inference time: {:.4f} seconds".format(test_correctness,
      time.time() - s))

classifier.eval()
s = time.time()
valid_correctness = evaluate(classifier, VALID_LOADER)
print("Validation Accuracy: {:.2f}%, inference time: {:.4f} seconds".format(
        valid_correctness, time.time() - s))


correct, total = 0, 0
LOADER, SET = TEST_LOADER, TEST_DATASET
# LOADER, SET = VALID_LOADER, VALID_DATASET
for batch_x, batch_y in LOADER:    
    pred = classifier(batch_x)
    pred_y = np.argmax(pred, axis=1)

    index = pred_y != batch_y

plt.figure(figsize=(16, 16))
for idx, (img, p, l, prob) in enumerate(
        zip(SET.x[index], pred_y[index], batch_y[index], pred[index]), 
        1):
    prob = np.exp(prob)
    prob = prob / np.sum(prob) * 100.
    top2 = heapq.nlargest(2, zip(prob, [_%10 for _ in range(1, len(prob)+1)]))
    
    plt.subplot((np.sum(index)//5)+1, 5, idx)
    plt.imshow(img.reshape(16, 16), "gray")
    plt.title("""pred: {}, truth: {}\n"{}"({:.2f}%)\n"{}"({:.2f}%)""".format(
                p+1, l+1, top2[0][1], top2[0][0], top2[1][1], top2[1][0]), 
              fontsize=15)
    plt.axis("off")

plt.show()
