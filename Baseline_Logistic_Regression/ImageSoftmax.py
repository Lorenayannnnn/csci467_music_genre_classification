"""Code for HW2 Problem 4: Neural Networks on MNIST.

We will use this notation:
    - B: size of batch
    - C: number of classes, i.e. NUM_CLASSES
    - D: size of inputs, i.e. INPUT_DIM
    - N: number of training examples
    - N_dev: number of dev examples
"""
import argparse
import copy
import sys
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image

OPTS = None

IMAGE_SHAPE = (218, 336)  # Size of MNIST images
NUM_CLASSES = 10  # Number of classes we are classifying over
#Note that NUMCOLS_PER_SECOND * HIDDEN_INPUT_DIM = 336
NUMCOLS_PER_SECOND = 6
HIDDEN_DIM = 56

#batch size = 32

class MusicDabber(nn.Module):
    def __init__(self):
        super(MusicDabber, self).__init__()
        self.RGBTransform = nn.Linear(3, 1)
        # self.RGBTransform = torch.nn.Parameter(torch.rand(3))
        # self.RGBTransformBias = torch.nn.Parameter(torch.rand(1))
        self.linear = nn.Linear(HIDDEN_DIM, NUM_CLASSES)
    
    #Input x is of dimension (200, 288, 432, 4)
    def forward(self, x):

        
        # print(f"x's shape is : {x.shape}")

        # print(torch.sum(x < 0))

        #Debug code
        # print(f"before transformation = {torch.sum(x[0] != x[1])}")
        # print(f"random number is {self.RGBTransform} and bias is {self.RGBTransformBias}")

        #End Debug code

        # print(f"float conversion = {torch.sum(x.float()[0] != x.float()[1])}")

        #Brightness is of dimension (200, 288, 432)
        brightness = self.RGBTransform(x)
        # print(brightness.shape)

        # print(torch.sum(brightness < 0))

        # print(f"Before relu but after linear transformation = {torch.sum(brightness[0] != brightness[1])}")

        #Brightness is now of dimension (200, 288, 432)

        # brightness = F.normalize(brightness, p = 1.0, dim = 0)
        # print(brightness.shape)
        # print(torch.mean(brightness[0]))
        # print(torch.mean(brightness[15]))
        # print(torch.sum(brightness < 0))

        brightness = F.tanh(brightness)

        # test = brightness.reshape(brightness.shape[0], brightness.shape[1], HIDDEN_DIM, NUMCOLS_PER_SECOND)
        # # print(f"after reshape = {brightness.shape}")
        # test = test.sum(dim = -1)
        # test = test.sum(dim = -2)
        # #The following is debug stuff
        # prist(brightness.shape)
        # for example in range(1):
        #     for col in range(0, 24):
        #         tempSum = 0
        #         for row in range(0, 288):
        #             for j in range(18):
        #                 tempSum += brightness[example][row][col * 18 + j]
        #         print(f"True sum for col {col} in row {example} is {tempSum}")
        #         print(f"tensor operations got sum {test[example][col]}")
        
        # firstExample = brightness[0]
        # tempSum = 0
        # for i in range(288):
        #     for j in range(18):
        #         tempSum += firstExample[i][j]

        # print(f"After brightness + relu = {torch.sum(brightness[0] != brightness[1])}")

        #Brightness is of dimension (200, 288, 24, 18)
        brightness = brightness.reshape(brightness.shape[0], brightness.shape[1], HIDDEN_DIM, NUMCOLS_PER_SECOND)

        # print(torch.sum(brightness < 0))
        
        # print(f"After reshaping = {torch.sum(x[0] != x[1])}")
        
        # print(f"after reshape = {brightness.shape}")
        brightness = brightness.mean(dim = -1)
        brightness = brightness.mean(dim = -2)
        #Brightness now has dimension (200, 24)


        # print(torch.sum(brightness < 0))
        # print(f"after sum = {brightness.shape}")
        output = self.linear(brightness)
        # print(f"output dim = {output.shape}")
        return output
    



def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=30):
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
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Stochastic gradient descent optimizer

    # Prepare the training dataset
    # Pytorch DataLoader expects a dataset to be a list of (x, y) pairs
    train_dataset = [(X_train[i,:], y_train[i]) for i in range(len(y_train))]

    # Simple version of early stopping: save the best model checkpoint based on dev accuracy
    best_dev_acc = -1
    best_checkpoint = None
    best_epoch = -1

    for t in range(num_epochs):


        train_num_correct = 0

        lossSum = 0

        # Training loop
        model.train()  # Set model to "training mode", e.g. turns dropout on if you have dropout layers
        for batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
                    # DataLoader automatically groups the data into batchse of roughly batch_size
                    # shuffle=True makes it so that the batches are randomly chosen in each epoch
            x_batch, y_batch = batch  # unpack batch, which is a tuple (x_batch, y_batch)
                                      # x_batch is tensor of size (B, D)
                                      # y_batch is tensor of size (B,)
            optimizer.zero_grad()  # Reset the gradients to zero
                                   # Recall how backpropagation works---gradients are initialized to zero and then accumulated
                                   # So we need to reset to zero before running on a new batch!
            logits = model(x_batch)   # tensor of size (B, C), each row is the logits (pre-softmax scores) for the C classes
                                      # For MNIST, C=10
            # print(logits)
            # print(y_batch)
            # return

            loss = loss_func(logits, y_batch)  # Compute the loss of the model output compared to true labels
            lossSum += loss
            loss.backward()  # Run backpropagation to compute gradients
            optimizer.step()  # Take a SGD step
                              # Note that when we created the optimizer, we passed in model.parameters()
                              # This is a list of all parameters of all layers of the model
                              # optimizer.step() iterates over this list and does an SGD update to each parameter

            # Compute running count of number of training examples correct
            preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
            train_num_correct += torch.sum(preds == y_batch).item()

        print(f"Loss for epoch {t} is {lossSum / (len(train_dataset) / batch_size)}")
        # Evaluate train and dev accuracy at the end of each epoch
        train_acc = train_num_correct / len(y_train)
        
        model.eval()  # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
        with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
            # print(X_dev.shape)
            # print(X_dev[0])
            # print(X_dev[1])
            # print(torch.sum(X_dev[0] != X_dev[1]))
            # print(torch.sum(X_dev[0] != X_dev[2]))
            dev_logits = model(X_dev)
            # print(f"dev_logits shape = {dev_logits.shape}")
            # print(dev_logits)
            dev_preds = torch.argmax(dev_logits, dim=1)
            # print(dev_preds)
            dev_acc = torch.mean((dev_preds == y_dev).float()).item()
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



def evaluate(model, X, y, name):
    """Measure and print accuracy of a predictor on a dataset."""
    model.eval()  # Set model to "eval mode", e.g. turns dropout off if you have dropout layers.
    with torch.no_grad():  # Don't allocate memory for storing gradients, more efficient when not training
        logits = model(X)  # tensor of size (N, 10)
        y_preds = torch.argmax(logits, dim=1)  # Choose argmax for each row (i.e., collapse dimension 1, hence dim=1)
        acc = torch.mean((y_preds == y).float()).item()
    print(f'    {name} Accuracy: {acc:.5f}')
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-1)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--num-epochs', '-T', type=int, default=30)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

#Tasks for tomorrow: 
'''
Start: Understand dataloader input types, etc
1. To extract the list of filenames from data_split.txt and store it into a tensor
2. To use dataloader to loop thru each of the files and train, etc
3. To then modify the neural.py to train BEFORE NOON
'''

#To read the train dev and test sets from the data split file
def read_from_data():
    f = open("./../data_split.txt")
    f.readline() # Burn train

    trainNames = list(map(lambda x: x[1:-1], f.readline()[1:-2].split(", ")))

    f.readline() # Burn dev

    devNames = list(map(lambda x: x[1:-1], f.readline()[1:-2].split(", ")))

    f.readline() # Burn test

    testNames = list(map(lambda x: x[1:-1], f.readline()[1:-2].split(", ")))

    return trainNames, devNames, testNames


# #For the CSV prediction
# class MyDataset(Dataset):
 
#   def __init__(self, file_name):
#     price_df=pd.read_csv(file_name)
#     labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
 
#     x=price_df.iloc[:,1:-1].values
# #     print(x)
    
# #     labelExtract = lambda x: i = i[:5] for i in x
#     y=np.array(list(map(lambda x: labels.index(x), price_df.iloc[:,-1].values)))
 
#     self.x_train=torch.tensor(x,dtype=torch.float32)
#     self.y_train=torch.tensor(y,dtype=torch.float32)
 
#   def __len__(self):
#     return len(self.y_train)
   
#   def __getitem__(self,idx):
#     return self.x_train[idx],self.y_train[idx]

# myDs = MyDataset("./../data/features_3_sec.csv")

# for batch in DataLoader(myDs, batch_size = 32, shuffle = True):
#     x_batch, y_batch = batch
#     print(x_batch.shape)
#     break

def main():
    # Set random seed, for reproducibility
    torch.manual_seed(0)

    # Read the data

    trainNames, devNames, testNames = read_from_data()

    data_address = "./../data/images_original/"

    labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    trainArrX = []
    trainArrY = []
    devArrX = []
    devArrY = []
    testArrX = []
    testArrY = []

    for trainExample in trainNames:
        im = Image.open(f"{data_address}{trainExample}")
        trainArrX += [np.asarray(im)]
        trainArrY += [labels.index(trainExample[:trainExample.find("/")])]
    
    for devExample in devNames:
        im = Image.open(f"{data_address}{devExample}")
        devArrX += [np.asarray(im)]
        devArrY += [labels.index(devExample[:devExample.find("/")])]

    for testExample in testNames:
        im = Image.open(f"{data_address}{testExample}")
        testArrX += [np.asarray(im)]
        testArrY += [labels.index(testExample[:testExample.find("/")])]
    
    #(700, 288, 432, 4)
    trainArrX = torch.tensor(np.stack(trainArrX))[:, 35:253, 54:390, :3].float()

    #(700)
    trainArrY = torch.tensor(np.array(trainArrY))

    #(200, 288, 432, 4)
    devArrX = torch.tensor(np.stack(devArrX))[:, 35:253, 54:390, :3].float()

    #(200)
    devArrY = torch.tensor(np.array(devArrY))

    #(99, 288, 432, 4)
    testArrX = torch.tensor(np.stack(testArrX))[:, 35:253, 54:390, :3].float()

    #(99)
    testArrY = torch.tensor(np.array(testArrY))


    print("Finish Data Collection")

            


    
    # print(trainArrX.shape)
    # print(trainArrY.shape)

    # print(trainArrX[0])

    # return

    model = MusicDabber()
    train(model, trainArrX, trainArrY, devArrX, devArrY, lr=OPTS.learning_rate,
          batch_size=OPTS.batch_size, num_epochs=OPTS.num_epochs)

    
    # Evaluate the model
    print('\nEvaluating final model:')
    train_acc = evaluate(model, trainArrX, trainArrY, 'Train')
    dev_acc = evaluate(model, devArrX, devArrY,  'Dev')
    PATH = "./models/"
    if OPTS.test:
        test_acc = evaluate(model, testArrX, testArrY, 'Test')
    torch.save(model.state_dict(), PATH + str(dev_acc)[2:6] + ".pt")
    


if __name__ == '__main__':
    OPTS = parse_args()
    main()

