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
from sklearn.metrics import confusion_matrix


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

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    net.to(dev)

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
            b_x = b_x.to(dev)
            b_y = Variable(entity['label']).reshape(len(entity['label']), )
            b_y = b_y.to(dev)

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


# Function for getting the key by values
def get_key(val):
    for key, value in class_name_dict.items():
        if val == value:
            return key

    return "key doesn't exist"


true_list = []
predict_list = []


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
        # filename = os.path.splitext(image)[0]
        # filename = os.path.basename(filename)
        # print('filename: ' + str(filename))
        class_name = os.path.dirname(image)
        class_name = os.path.basename(class_name)
        true_list.append(class_name)
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

        predict_list.append(get_key(index))

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

    print(true_list)
    print(predict_list)

    overall_accuracy = class_name_accurate_times / compare_times
    print('OVER')
    print(str(overall_accuracy * 100) + '%')


# test_and_predict()


def confusion_matrix_calculator():
    c = confusion_matrix(true_list, predict_list)
    print(c)


# confusion_matrix_calculator()


# Revise from chenzhi

EPOCH = 20
LEARNING_RATE = 0.00001


def load_model():
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn.pkl'.format(EPOCH, LEARNING_RATE)))
    return cnn


def test_model(cnn):
    cnn.eval()

    total = 0
    correct = 0
    overall_accuracy = 0.0
    classification_accurate_times_per_class = 0
    class_compare_times = 0
    previous_class_index = 0
    previous_class_name = ''
    accuracy_array = [0, 0, 0, 0, 0]

    test_set = imgdata.DefaultTestSet()
    test_loader = Data.DataLoader(dataset=test_set)
    conf_arr = np.zeros((5, 5), dtype=np.int)  # create a 5*5 matrix with all values 0
    for index, entity in enumerate(test_loader):
        b_x = Variable(entity['imNorm'])
        b_y = Variable(entity['label']).numpy().squeeze()
        b_y = int(b_y)
        pred_y = cnn.predict(b_x)

        # print(pred_y)
        print(b_y)
        total += 1
        # For total prediction
        if b_y == pred_y:
            correct += 1
        conf_arr[b_y][pred_y] += 1

        # For every class accuracy
        if b_y == pred_y:
            accuracy_array[b_y] += 1


    print('Overall accuracy: ' + str(correct / total * 100) + '%')
    print('Correct predicted items: ' + str(correct))
    print('Total predicted items: ' + str(total))
    print('Confusion matrix: \n' + str(conf_arr))
    print("-----------------")
    for index in range(len(accuracy_array)):
        print('Accuracy of ' + str(get_key(index)) + ' is: ' + str(accuracy_array[index] * 100 / 10) + '%')


if __name__ == "__main__":
    test_model(load_model())
