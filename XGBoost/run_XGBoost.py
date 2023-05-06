from torch.utils.data import DataLoader, Dataset
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from xgboost import XGBClassifier


# For the CSV prediction
class MyDataset(Dataset):

    def __init__(self, file_name, nameList):
        price_df = pd.read_csv(file_name)
        labels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

        # Edits the nameList such that the formatting is right (from blues/blues00001.png -> blues00001)
        nameList = list(map(lambda x: x[x.index("/") + 1:x.index(".")], nameList))
        nameList = list(map(lambda x: x[:x.index("0")] + "." + x[x.index("0"):], nameList))

        # An array of booleans indicating whether or not the values were in the nameList
        nameIsIn = [(i[:-6] in nameList) for i in price_df.iloc[:, 0].values]

        # Notice the last index so that we can filter out the entries where we don't want to include things
        x = price_df.iloc[:, 2:-1].values[nameIsIn]

        y = np.array(list(map(lambda x: labels.index(x), price_df.iloc[:, -1].values)))[nameIsIn]

        # self.x=F.normalize(torch.tensor(x,dtype=torch.float32), p = 1.0, dim = 1)
        # (6992, 58)

        # print(type(x))
        mean = np.mean(x, axis=0)
        # print(mean.size)
        std = np.std(x, axis=0)
        # print(f"original = {x.shape}")
        # print(f"new = {((np.subtract(x, mean))).shape}")
        # print(f"new mean = {np.subtract(x, mean).mean(axis=0)}")
        # print(f"new std = {np.divide(x, std).std(axis=0)}")
        # print(f"new std shape = {np.divide(x, std).shape}")
        # print(f"std = {std}")
        self.x = torch.tensor(np.divide(np.subtract(x, mean), std))
        # self.x = F.normalize(torch.tensor(x, dtype=torch.float32), p=2.0, dim=1)
        # self.x = x - mean
        # (6992,)
        self.y = torch.tensor(y)

        print(f"self.x shape = {self.x.shape}")
        print(f"self.y shape = {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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


trainNames, devNames, testNames = read_from_data()

trainDs = MyDataset("./../data/features_3_sec.csv", trainNames)
devDs = MyDataset("./../data/features_3_sec.csv", devNames)
testDs = MyDataset("./../data/features_3_sec.csv", testNames)

# bst = XGBClassifier(n_estimators=100, max_depth=7, learning_rate=.1, booster="gbtree", tree_method="exact",
#                     objective='multi:softmax')

bst = XGBClassifier(objective="binary:logistic")

# df = pd.DataFrame(testDs.x)
# df = pd.DataFrame(testDs.y)

# print(df.mean(axis=0))

# fit model
bst.fit(trainDs.x, trainDs.y)
# make predictions
preds = bst.predict(testDs.x)

print(sum(np.array(preds) == np.array(testDs.y)) / testDs.y.shape[0])
