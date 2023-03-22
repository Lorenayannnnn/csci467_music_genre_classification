import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import cnn

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
test_dataset = datasets.ImageFolder(root_dir + 'test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# call our best model
model = cnn.ConvNet(num_classes=num_classes, pad=3, kernel=5, h1=128, stride=2)
model.load_state_dict(torch.load('./checkpoints/best/epoch140.pt'))
model.eval()
model.cuda()

# compute accuracy on the test set with error of each class
with torch.no_grad():
    correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct1 = 0
    total1 = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(labels.size(0)):
            total[labels[i]] += 1
            if labels[i] == predicted [i]:
                correct[labels[i]] += 1
        total1 += labels.size(0)
        correct1 += (predicted == labels).sum().item()

    for i in range(10):
        print(correct[i])
        print(total[i])
        print('accuracy for class {} is: {}'.format(str(i+1), correct[i]/total[i]))
    print(
        'Accuracy of the network on the test images: {} %'.format(100 * correct1 / total1))


