from Model_Training import CNN
import torch
import imgdata
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import torch.nn as nn


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


EPOCH = 20
LEARNING_RATE = 0.0001

def load_model(netname, n_epochs, learning_rate):
    if netname == "cnn":
        cnn = CNN()
        print(netname + "_{0}_{1}.pkl".format(n_epochs,learning_rate))
        cnn.load_state_dict(torch.load(netname + "_{0}_{1}.pkl".format(n_epochs,learning_rate)))
        return cnn
    elif netname == "resnet":
        resnet = models.resnet18()
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 5)
        resnet.load_state_dict(torch.load(netname + "_{0}_{1}.pkl".format(n_epochs,learning_rate)))
        return resnet
    elif netname == "vgg":
        vgg = models.vgg16(pretrained=True)
        num_ftrs = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(num_ftrs, 5)
        vgg.load_state_dict(torch.load(netname + "_{0}_{1}.pkl".format(n_epochs,learning_rate)))
        return vgg
    elif netname == "alexnet":
        alexnet = models.alexnet(pretrained=True)
        num_ftrs = alexnet.classifier[6].in_features
        alexnet.classifier[6] = nn.Linear(num_ftrs, 5)
        alexnet.load_state_dict(torch.load(netname + "_{0}_{1}.pkl".format(n_epochs,learning_rate)))
        return alexnet

def test_model(net, netname, test_loader,):
    net.eval()
    total = 0
    correct = 0

    conf_arr = np.zeros((5, 5), dtype=np.int)  # create a 5*5 matrix with all values 0

    accuracy_array = [0, 0, 0, 0, 0]
    with torch.no_grad():
        for index, entity in enumerate(test_loader):
            b_x = Variable(entity['imNorm'])
            b_y = Variable(entity['label']).numpy().squeeze()
            b_y = int(b_y)
            if netname == "cnn":
                outputs= net.predict(b_x)
            else:
                outputs = net(b_x)
            _, pred_y = torch.max(outputs.data, 1)

            print('---------------------------------------')
            print('Prediction: ' + str(pred_y))
            print('Actual: ' + str(b_y))
            print('---------------------------------------')

            total += 1
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
    test_set = imgdata.DefaultTestSet()
    test_loader = Data.DataLoader(dataset=test_set)
    # netname "resnet" for ResNet, "vgg" for VGG, "alexnet" for AlexNet
    # models are saved as modelname_epochs_learningrate.pkl
    netname="cnn"
    model =load_model(netname, EPOCH,LEARNING_RATE)
    test_model(model, netname,test_loader)
