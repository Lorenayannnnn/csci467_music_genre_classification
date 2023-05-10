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
test_dataset = datasets.ImageFolder(root_dir + 'dev/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# call our best model
model = cnn.ConvNet_model6(num_classes=num_classes, pad=1, kernel=10, h1=32, stride=2)
model.load_state_dict(torch.load('./checkpoints/final_model6/epoch23.pt'))
model.eval()
model.cuda()

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
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

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