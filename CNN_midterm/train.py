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

# use GPU to speed up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load datasets
transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])
train_dataset = datasets.ImageFolder(root_dir + 'train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

dev_dataset = datasets.ImageFolder(root_dir + 'dev/', transform=transform)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

# initialize the model
model = cnn.ConvNet(num_classes=num_classes, pad=3, kernel=5, h1=128, stride=2)
model.cuda()

# set seed
torch.manual_seed(42)

# Set Loss function
criterion = nn.CrossEntropyLoss()

# Set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)

total_step = len(train_loader)


current_accuracy = 0
best_accuracy = 0
best_train = []
best_epoch = 0

for epoch in tqdm(range(num_epochs)):

    # Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    if True:
        # compute training accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(
                'Accuracy of the network on the train images at {} epochs: {} %'.format(str(epoch+1), 100 * correct / total))
            best_train.append(100 * correct / total ) 

        # compute dev accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in dev_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the dev images at {} epochs: {} %'.format(str(epoch+1), 100 * correct / total))

            # save checkpoints for better performance
            current_accuracy = 100 * correct / total
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_epoch = epoch+1
        
                PATH = './checkpoints/best/epoch{}.pt'.format(epoch+1)
                torch.save(model.state_dict(), PATH)

print('Best dev Accuracy: {} %'.format(best_accuracy))
print('Best train: {} %'.format(best_train[best_epoch-1]))
print('Best epoch: {}'.format(best_epoch))
