import numpy as np
import imgdata
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            # 3*250*250 -> 64*125*125
            nn.Conv2d(in_channels=IN_CHANNELS, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64*125*125 -> 64*62*62
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

        self.conv2 = nn.Sequential(
            # 64*62*62 -> 64*62*62
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64*62*62 -> 64*30*30
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

        self.fc = nn.Linear(57600, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = softmax(x)
        x = self.softmax(x)
        return x
        # return torch.max(x, 1)[1].data.numpy().squeeze()


def softmax(input):
    input.exp_()
    sum = torch.cumsum(input, dim=1)
    for i in range(len(input)):
        input[i, :] /= sum[i, 4]
    return input


def plot(data, label):
    x = np.arange(1, len(data) + 1)

    plt.plot(x, data, color="red", label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title("Evolution of the " + label)
    plt.savefig(label + ".jpg")
    plt.show()


def train_model(net, net_name, n_epochs, batch_size, learning_rate):
    print("n_epochs=", n_epochs)
    print("batch_size=", batch_size)
    print("learning_rate=", learning_rate)
    print("=" * 30)




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.max_memory_allocated()

    print(device)
    net.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    train_set = imgdata.DefaultTrainSet()
    train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

    train_loss = []
    train_accuracy = []

    for epoch in range(n_epochs):
        total_train_loss = 0
        correct = 0

        for _, entity in enumerate(train_loader):
            b_x = Variable(entity['imNorm'])
            b_x = b_x.to(device)
            b_y = Variable(entity['label']).reshape(len(entity['label']))
            b_y = b_y.to(device)

            output = net(b_x)
            loss = loss_func(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == b_y).sum().item()

        avg_loss = total_train_loss / len(train_loader)
        accuracy = 100 * correct / len(train_set)
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch + 1, n_epochs, avg_loss, accuracy))

        train_loss.append(avg_loss)
        train_accuracy.append(accuracy)

    torch.save(net.state_dict(), net_name + ".pkl")

    plot(train_loss, "training loss")
    plot(train_accuracy, "training accuracy")


def test_model(net, net_name):
    net.load_state_dict(torch.load(net_name + '.pkl'))
    net.eval()

    test_set = imgdata.DefaultTestSet()
    test_loader = Data.DataLoader(dataset=test_set)

    correct = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    with torch.no_grad():
        for entity in test_loader:
            images = Variable(entity['imNorm'])
            images=images.to(device)
            labels = Variable(torch.squeeze(entity['label'])).to(device)
            # outputs = net.predict(images)
            outputs = net(images)
            pred_y = torch.max(outputs.data, 1)[1]
            correct += (pred_y == labels).sum().item()

    accuracy = 100 * correct / len(test_set)
    print("Test accuracy: {:.2f}%".format(accuracy))


seed = 42
torch.manual_seed(seed)

EPOCH = 20
BATCH_SIZE = 5
LEARNING_RATE = 0.01
IN_CHANNELS = 3

cnn = CNN()

# ResNet
resnet18 = models.resnet18(pretrained=True)
# Modify pretrained model
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 5)

# VGG
vgg16 = models.vgg16(pretrained=True)
num_ftrs = vgg16.classifier[6].in_features
vgg16.classifier[6] = nn.Linear(num_ftrs, 5)

# AlexNet
alexnet = models.alexnet(pretrained=True)
num_ftrs = alexnet.classifier[4].in_features
alexnet.classifier[4] = nn.Linear(num_ftrs, 1024)
alexnet.classifier[6] = nn.Linear(1024, 5)

# train_model(vgg16, "vgg16", EPOCH, BATCH_SIZE, LEARNING_RATE)

test_model(vgg16, "vgg16")
