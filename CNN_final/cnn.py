import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import tqdm
from torchvision import datasets
from torch.nn import functional as F


class ConvNet_model1(nn.Module):

    def __init__(self, num_classes, pad, kernel, h1, stride):
        super(ConvNet_model1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel, padding=pad, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, padding=pad, stride = stride)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel, padding=pad, stride = stride)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel, padding=pad, stride = stride)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(256, h1)
        self.drop2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(h1, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv4(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.drop1(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop2(out)

        out = self.fc2(out)

        return out


class ConvNet_model2(nn.Module):

    def __init__(self, num_classes, pad, kernel, h1, stride):
        super(ConvNet_model2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel, padding=pad, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, padding=pad, stride = stride)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel, padding=pad, stride = stride)
        
        self.drop1 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(768, h1)
        self.drop2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(h1, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.drop1(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop2(out)

        out = self.fc2(out)

        return out


class ConvNet_model3(nn.Module):

    def __init__(self, num_classes, pad, kernel, h1, stride):
        super(ConvNet_model3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel, padding=pad, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, padding=pad, stride = stride)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel, padding=pad, stride = stride)
        
        self.drop1 = nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(768, h1)
        self.drop2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(h1, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.drop1(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop2(out)

        out = self.fc2(out)

        return out


class ConvNet_model4(nn.Module):

    def __init__(self, num_classes, pad, kernel, h1, stride):
        super(ConvNet_model4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel, padding=pad, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, padding=pad, stride = stride)
        
        
        self.drop1 = nn.Dropout(0.1)
        
        self.fc1 = nn.Linear(8320, h1)
        self.drop2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(h1, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)


        out = self.drop1(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop2(out)

        out = self.fc2(out)

        return out


class ConvNet_model5(nn.Module):

    def __init__(self, num_classes, pad, kernel, h1, stride):
        super(ConvNet_model5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel, padding=pad, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, padding=pad, stride = stride)
        
        
        self.drop1 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(8320, h1)
        self.drop2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(h1, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)


        out = self.drop1(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop2(out)

        out = self.fc2(out)

        return out


class ConvNet_model6(nn.Module):

    def __init__(self, num_classes, pad, kernel, h1, stride):
        super(ConvNet_model6, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel, padding=pad, stride = stride)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel, padding=pad, stride = stride)
        
        
        self.drop1 = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(6688, h1)
        self.drop2 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(h1, num_classes)


    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)

        out = self.conv2(out)
        out = F.max_pool2d(out, kernel_size=2)
        out = F.relu(out)


        out = self.drop1(out)

        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = F.relu(out)
        out = self.drop2(out)

        out = self.fc2(out)

        return out