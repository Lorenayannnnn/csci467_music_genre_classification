import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import cnn
import confusion_matrix
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

from torchvision import datasets

my_dict = {'blues':0, 'classical':1, 'country':2, 'disco':3, 'hiphop':4, 'jazz':5,
 'metal':6, 'pop':7, 'reggae':8, 'rock':9}

def emsemble(pred1, pred2, pred3, pred4, pred5, pred6, labels):
    # Stack the predictions from all five models along a new dimension
    stacked_preds = torch.stack([pred1, pred2, pred3, pred4, pred5, pred6], dim=1)
    pred = []

    for preds_per_example in stacked_preds:
        
        # find the unique values and their counts
        unique_values, counts = torch.unique(preds_per_example, return_counts=True)
        
        # find the index of the maximum count
        max_idx = torch.argmax(counts)

        # there is a majority vote
        if counts[max_idx] >= 3:
            pred.append(unique_values[max_idx])
        # use the best model
        else:
            pred.append(preds_per_example[0].item())
            
    pred = torch.tensor(pred)
    
    return pred


# Define hyperparamteres
batch_size = 64
num_classes = 10
learning_rate = 0.0005
num_epochs = 150
root_dir = '../../Data/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
                                transforms.ToTensor()
                                ])

# load test data
test_dataset = datasets.ImageFolder(root_dir + 'test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# call our 5 CNN models for emsembling
model1 = cnn.ConvNet_model1(num_classes=num_classes, pad=3, kernel=5, h1=128, stride=2)
model1.load_state_dict(torch.load('./checkpoints/final_model1/best.pt'))
model1.eval()
model1.cuda()

model2 = cnn.ConvNet_model2(num_classes=num_classes, pad=1, kernel=5, h1=128, stride=2)
model2.load_state_dict(torch.load('./checkpoints/final_model2/best.pt'))
model2.eval()
model2.cuda()

model3 = cnn.ConvNet_model3(num_classes=num_classes, pad=1, kernel=5, h1=128, stride=2)
model3.load_state_dict(torch.load('./checkpoints/final_model3/best.pt'))
model3.eval()
model3.cuda()

model4 = cnn.ConvNet_model4(num_classes=num_classes, pad=1, kernel=5, h1=128, stride=2)
model4.load_state_dict(torch.load('./checkpoints/final_model4/best.pt'))
model4.eval()
model4.cuda()

model5 = cnn.ConvNet_model5(num_classes=num_classes, pad=1, kernel=5, h1=128, stride=2)
model5.load_state_dict(torch.load('./checkpoints/final_model5/best.pt'))
model5.eval()
model5.cuda()

model6 = cnn.ConvNet_model6(num_classes=num_classes, pad=1, kernel=10, h1=32, stride=2)
model6.load_state_dict(torch.load('./checkpoints/final_model6/best.pt'))
model6.eval()
model6.cuda()

idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

# compute accuracy on the test set with error of each class
with torch.no_grad():
    correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct1 = 0
    total1 = 0
    confusion_mat = np.zeros((10, 10))
    pred_labels = []
    true_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs1 = model1(images)
        _, pred1 = torch.max(outputs1.data, 1)
        outputs2 = model2(images)
        _, pred2 = torch.max(outputs2.data, 1)
        outputs3 = model3(images)
        _, pred3 = torch.max(outputs3.data, 1)
        outputs4 = model4(images)
        _, pred4 = torch.max(outputs4.data, 1)
        outputs5 = model5(images)
        _, pred5 = torch.max(outputs5.data, 1)
        outputs6 = model6(images)
        _, pred6 = torch.max(outputs6.data, 1)

        predicted = emsemble(pred1, pred2, pred3, pred4, pred5, pred6, labels)
        predicted = predicted.to(device)
        # print(predicted)

        for i in range(labels.size(0)):
            x = labels[i].item()
            y = predicted[i].item()
            true_labels.append(idx_to_class[x])
            pred_labels.append(idx_to_class[y])
            confusion_mat[x][y] += 1
            total[labels[i]] += 1
            if labels[i] == predicted[i]:
                correct[labels[i]] += 1
        total1 += labels.size(0)
        # print(predicted.device, labels.device)
        correct1 += (predicted == labels).sum().item()
    

    for i in range(10):
        print(correct[i])
        print(total[i])
        print('accuracy for class {} is: {}'.format(idx_to_class[i], correct[i]/total[i]))
    print(
        'Accuracy of the network on the test images: {} %'.format(100 * correct1 / total1))

    print(confusion_mat)

    for i in range(10):
        print(idx_to_class[i], end=' ')
    print()

    confusion_matrix.report_classification_accuracy(true_labels, pred_labels)