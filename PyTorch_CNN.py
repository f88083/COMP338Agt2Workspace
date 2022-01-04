import os

import joblib as joblib
import numpy as np
import imgdata
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image
import glob


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
        x = softmax(x)
        return torch.max(x, 1)[1].data.numpy().squeeze()


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


def train_model(net, n_epochs, batch_size, learning_rate):
    print("n_epochs=", n_epochs)
    print("batch_size=", batch_size)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    train_set = imgdata.DefaultTrainSet()
    train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

    train_loss = []
    train_accuracy = []

    for epoch in range(n_epochs):
        # train_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
        total_train_loss = 0
        correct = 0

        for _, entity in enumerate(train_loader):
            b_x = Variable(entity['imNorm'])
            b_y = Variable(entity['label']).reshape(len(entity['label']), )

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

    torch.save(net.state_dict(), "cnn.pkl")

    plot(train_loss, "training loss")
    plot(train_accuracy, "training accuracy")


def test_model():
    model = CNN()
    # num_ftrs = model.fc.in_features
    # if num_ftrs != 5:
    #     model.fc = nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load('cnn.pkl'))
    model.eval()

    test_set = imgdata.DefaultTestSet()
    test_loader = Data.DataLoader(dataset=test_set)

    correct = 0

    with torch.no_grad():
        for entity in test_loader:
            images = Variable(entity['imNorm'])
            labels = Variable(torch.squeeze(entity['label']))
            # outputs = model.predict(images)
            outputs = model(images)
            pred_y = torch.max(outputs.data, 1)[1]
            correct += (pred_y == labels).sum().item()

    accuracy = 100 * correct / len(test_set)
    print("Test accuracy: {:.2f}%".format(accuracy))


seed = 42
torch.manual_seed(seed)

EPOCH = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.01
IN_CHANNELS = 3

cnn = CNN()
# train_model(cnn, EPOCH, BATCH_SIZE, LEARNING_RATE)


# load the model from disk
loaded_model = CNN()
loaded_model.load_state_dict(torch.load('cnn.pkl'))
loaded_model.eval()
# Load an image
# img = Image.open('data/COMP338_Assignment2_Dataset/Test/airplanes/0005.jpg')
#
# output = loaded_model(img)


# ---------------------------------------------------
# Create the preprocessing transformation here
transform = transforms.ToTensor()
# load your image(s)
img = Image.open('data/COMP338_Assignment2_Dataset/Test/keyboard/0075.jpg')

img_path = 'data/COMP338_Assignment2_Dataset/Test/*/*.jpg'

class_name_dict = {
    'airplanes': 0,
    'cars': 1,
    'dog': 2,
    'faces': 3,
    'keyboard': 4
}


# test_model()

def test_and_predict():
    class_name_accurate_times = 0
    compare_times = 0
    overall_accuracy = 0.0
    classification_accurate_times_per_class = 0
    class_compare_times = 0
    previous_class_index = 0
    previous_class_name = ''

    for image in glob.glob(img_path):
        class_name = os.path.dirname(image)
        class_name = os.path.basename(class_name)

        current_class_name_index = class_name_dict.get(class_name)
        # print('current class name : ' + str(class_name))
        # print('current class name index : ' + str(current_class_name_index))

        img = Image.open(image)
        # Transform
        input = transform(img)

        # unsqueeze batch dimension, in case you are dealing with a single image
        input = input.unsqueeze(0)

        # Get prediction
        output = loaded_model(input)

        index = torch.argmax(output)

        if index.item() == current_class_name_index:
            class_name_accurate_times += 1

        if previous_class_index == current_class_name_index:
            if index.item() == current_class_name_index:
                classification_accurate_times_per_class += 1
            class_compare_times += 1
            previous_class_name = class_name
        else:
            print('Class ' + str(previous_class_name) + ' accuracy is: ' + str(
                classification_accurate_times_per_class / class_compare_times * 100) + '%')
            classification_accurate_times_per_class = 0
            class_compare_times = 0
            previous_class_index = current_class_name_index
            if index.item() == current_class_name_index:
                classification_accurate_times_per_class += 1
            class_compare_times += 1

        compare_times += 1

    # For the last class
    print('Class ' + str(previous_class_name) + ' accuracy is: ' + str(
        classification_accurate_times_per_class / class_compare_times * 100) + '%')


    overall_accuracy = class_name_accurate_times / compare_times
    print('OVER')
    print(str(overall_accuracy * 100) + '%')


test_and_predict()

# ---------------------------------------------------------------
# # 获取图像路径
# filelist = glob.glob('data/COMP338_Assignment2_Dataset/Test/*.jpg')
#
# # 打开图像open('frame_path')--》转换为灰度图convert('L')--》缩放图像resize((width, height)) --》合并文件夹中的所有图像为一个numpy array
# x = np.array([np.array(Image.open(frame).convert('L').resize((128, 128))) for frame in filelist])
#
# # 用torch.from_numpy这个方法将numpy类转换成tensor类
# x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
#
# # 扩充数据维度
# x = Variable(torch.unsqueeze(x, dim=1).float(), requires_grad=False)


# --------------------------------------------------------------
# trans = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(32),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
#     ])
# input = trans(img)
# input = input.view(1, 3, 32,32)
# output = loaded_model(input)

# -------------------------------------
# # reshape sample to (batch-size x width x height) but batch-size is 1 because you probably want to predict just one image at a time in real-life usage
# img = torch.reshape(1, img.size(0), img.size(1))
#
# prediction = loaded_model(img)
#
# print(prediction)
