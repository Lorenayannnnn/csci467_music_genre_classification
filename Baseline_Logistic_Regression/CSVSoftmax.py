
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

import sklearn.preprocessing as skp

# For normalizing
scaler = skp.StandardScaler()

OPTS = None

IMAGE_SHAPE = (218, 336)  # Size of MNIST images
NUM_CLASSES = 10  # Number of classes we are classifying over
# Note that NUMCOLS_PER_SECOND * HIDDEN_INPUT_DIM = 336
INPUT_DIM = 57

HIDDEN_DIM1 = 512
HIDDEN_DIM2 = 256
HIDDEN_DIM3 = 128
HIDDEN_DIM4 = 64


# HIDDEN_DIM4 = 64


# batch size = 32

class MusicDabber(nn.Module):
    def __init__(self):
        super(MusicDabber, self).__init__()

        # Softmax regression
        # self.linear = nn.Linear(INPUT_DIM, NUM_CLASSES)

        # 2 Layer NN
        # self.linear = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        # self.linear2 = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

        # 3 Deep NN
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM1)
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2)
        self.dropout2 = nn.Dropout(0.45)
        self.linear3 = nn.Linear(HIDDEN_DIM2, HIDDEN_DIM3)
        self.dropout3 = nn.Dropout(0.325)
        self.linear4 = nn.Linear(HIDDEN_DIM3, HIDDEN_DIM4)
        self.dropout4 = nn.Dropout(0.2)
        self.linear5 = nn.Linear(HIDDEN_DIM4, NUM_CLASSES)

        # self.dropout = nn.Dropout(0.1)

    # Input x is of dimension (200, 288, 432, 4)
    def forward(self, x):
        # Softmax Regression
        # output = self.linear(x)

        # 2 layer NN
        # output1 = self.linear(x)
        # # activation = F.relu(self.dropout(output1))
        # activation = F.relu(output1)
        # output = self.linear2(activation)

        # 3 Deep NN
        output1 = self.linear1(x)
        activation1 = F.relu(self.dropout1(output1))

        output2 = self.linear2(activation1)
        activation2 = F.relu(self.dropout2(output2))

        output3 = self.linear3(activation2)
        activation3 = F.relu(self.dropout3(output3))

        output4 = self.linear4(activation3)
        activation4 = F.relu(self.dropout4(output4))

        output5 = self.linear5(activation4)

        output = output5

        return output


def train(model, trainDs, devDs, lr=1e-1, batch_size=32, num_epochs=30, weight_decay=1e-5):
    """Run the training loop for the model.

    All of this code is highly generic and works for any model that does multi-class classification.

    Args:
        model: A nn.Module model, must take in inputs of size (B, D)
               and output predictions of size (B, C)
        X_train: Tensor of size (N, D)
        y_train: Tensor of size (N,)
        X_dev: Tensor of size (N_dev, D). Used for early stopping.
        y_dev: Tensor of size (N_dev,). Used for early stopping.
        lr: Learning rate for SGD
        batch_size: Desired batch size.
        num_epochs: Number of epochs of SGD to run
    """
    start_time = time.time()
    loss_func = nn.CrossEntropyLoss()  # Cross-entropy loss is just softmax regression loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Stochastic gradient descent optimizer

    # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_acc = -1
    best_checkpoint = None
    best_epoch = -1

    for t in range(num_epochs):

        train_num_correct = 0

        lossSum = 0

        # Training loop
        model.train()  # Set model to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(trainDs, batch_size=batch_size, shuffle=True):
            # DataLoader automatically groups the data into batchse of roughly batch_size
            # shuffle=True makes it so that the batches are randomly chosen in each epoch
            x_batch, y_batch = batch  # unpack batch, which is a tuple (x_batch, y_batch)
            # x_batch is tensor of size (B, D)
            # y_batch is tensor of size (B,)
            optimizer.zero_grad()  # Reset the gradients to zero
            # Recall how backpropagation works---gradients are initialized to zero and then accumulated
            # So we need to reset to zero before running on a new batch!
            logits = model(
                x_batch)  # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
            # For MNIST, C=10
            # print(logits)
            # print(y_batch)
            # return

            loss = loss_func(logits, y_batch)  # Compute the loss of the model output compared to true labels
            lossSum += loss.item()
            loss.backward()  # Run backpropagation to compute gradients
            optimizer.step()  # Take a SGD step
            # Note that when we created the optimizer, we passed in model.parameters()
            # This is a list of all parameters of all layers of the model
            # optimizer.step() iterates over this list and does an SGD update to each parameter

            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
            train_num_correct += torch.sum(preds == y_batch).item()

        print(f"Loss for epoch {t} is {lossSum / (len(trainDs) / batch_size)}")
        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(trainDs)

        model.eval()  # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
            # print(X_dev.shape)
            # print(X_dev[0])
            # print(X_dev[1])
            # print(torch.sum(X_dev[0] != X_dev[1]))
            # print(torch.sum(X_dev[0] != X_dev[2]))
            dev_logits = model(devDs.x)
            # print(f"dev_logits shape = {dev_logits.shape}")
            # print(dev_logits)
            dev_preds = torch.argmax(dev_logits, dim=1)
            # print(dev_preds)
            dev_acc = torch.mean((dev_preds == devDs.y).float()).item()
            if dev_acc > best_dev_acc:
                # Save this checkpoint if it has best dev accuracy so far
                best_dev_acc = dev_acc
                best_checkpoint = copy.deepcopy(model.state_dict())
                best_epoch = t
        # print(f'Epoch {t: <2}: dev_acc={dev_acc:.5f}')
        print(f'Epoch {t: <2}: train_acc={train_acc:.5f}, dev_acc={dev_acc:.5f}')

    # Set the model parameters to the best checkpoint across all epochs
    model.load_state_dict(best_checkpoint)
    end_time = time.time()
    print(f'Training took {end_time - start_time:.2f} seconds')
    print(f'\nBest epoch was {best_epoch}, dev_acc={best_dev_acc:.5f}')


def evaluate(model, inputDs, name):
    """Measure and print accuracy of a predictor on a dataset."""
    model.eval()  # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
    with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
        logits = model(inputDs.x)  # tensor of size (N, 10)
        y_preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
        acc = torch.mean((y_preds == inputDs.y).float()).item()
    print(f'    {name} Accuracy: {acc:.5f}')
    return acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-1)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--num-epochs', '-T', type=int, default=30)
    parser.add_argument('--weight_decay', '-w', type=float, default=1e-5)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


# To read the train dev and test sets from the data split file
def read_from_data():
    f = open("./../data_split.txt")
    f.readline()  # Burn train

    trainNames = list(map(lambda x: x[1:-1], f.readline()[1:-2].split(", ")))

    f.readline()  # Burn dev

    devNames = list(map(lambda x: x[1:-1], f.readline()[1:-2].split(", ")))

    f.readline()  # Burn test

    testNames = list(map(lambda x: x[1:-1], f.readline()[1:-2].split(", ")))

    return trainNames, devNames, testNames


# For the CSV prediction
class MyDataset(Dataset):

    def __init__(self, file_name, nameList):
        audio_feature_df = pd.read_csv(file_name)

        labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

        # Edits the nameList such that the formatting is right (from blues/blues00001.png -> blues00001)
        nameList = list(map(lambda x: x[x.index("/") + 1:x.index(".")], nameList))
        nameList = list(map(lambda x: x[:x.index("0")] + "." + x[x.index("0"):], nameList))

        # An array of booleans indicating whether or not the values were in the nameList
        nameIsIn = [(i[:-6] in nameList) for i in audio_feature_df.iloc[:, 0].values]

        # Notice the last index so that we can filter out the entries where we don't want to include things
        x = audio_feature_df.iloc[:, 2:-1].values[nameIsIn]
        x_df = pd.DataFrame(x)

        x = scaler.fit_transform(x_df)

        y = np.array(list(map(lambda tmp_x: labels.index(tmp_x), audio_feature_df.iloc[:, -1].values)))[nameIsIn]

        self.x = torch.tensor(x, dtype=torch.float32)

        # (6992,)
        self.y = torch.tensor(y)

        print(f"self.x shape = {self.x.shape}")
        print(f"self.y shape = {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    # Set random seed, for reproducibility
    torch.manual_seed(12)

    # Read the data

    trainNames, devNames, testNames = read_from_data()

    trainDs = MyDataset("./../data/features_3_sec.csv", trainNames)
    devDs = MyDataset("./../data/features_3_sec.csv", devNames)
    testDs = MyDataset("./../data/features_3_sec.csv", testNames)

    print("Finish Data Collection")

    # print(len(trainDs.x))
    # return

    model = MusicDabber()
    train(model, trainDs, devDs, lr=OPTS.learning_rate,
          batch_size=OPTS.batch_size, num_epochs=OPTS.num_epochs, weight_decay=OPTS.weight_decay)

    # Evaluate the model
    print('\nEvaluating final model:')
    train_acc = evaluate(model, trainDs, 'Train')
    dev_acc = evaluate(model, devDs, 'Dev')
    PATH = "./models/"
    if OPTS.test:
        test_acc = evaluate(model, testDs, 'Test')
    # torch.save(model.state_dict(), PATH + str(dev_acc)[2:6] + ".pt")


if __name__ == '__main__':
    OPTS = parse_args()
    main()
