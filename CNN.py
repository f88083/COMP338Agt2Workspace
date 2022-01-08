import numpy as np
import torch
import torch.nn as nn
import imgdata
import skimage
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F

EPOCH = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IN_CHANNELS = 3

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=IN_CHANNELS, out_channels=64, kernel_size=(7,7), stride=(2,2), padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)#64*62*62
        )
        #64*42*42
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),#64*62*62
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=0)#64*30*30
        )

        self.fc1=nn.Linear(57600, 5)


    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        return x

    def predict(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.nn.functional.softmax(x)
        return torch.argmax(x).numpy()

# def softmax(input):
#     input.exp_()
#     sum=torch.cumsum(input, dim=1)
#     for i in range(len(input)):
#         input[i, :] /= sum[i, 4]
#     return input

def train_model():
    cnn=CNN()
    optimizer=torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    loss_func=nn.CrossEntropyLoss()

    train_set=imgdata.DefaultTrainSet()
    Loss = []
    for epoch in range(EPOCH):
        train_loader = Data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False)
        running_loss = 0
        for _, entity in enumerate(train_loader):
            # print(step)
            b_x = Variable(entity['imNorm'])
            b_y = Variable(entity['label']).reshape(len(entity['label']),)
            # print(b_y)

            output=cnn(b_x)
            loss=loss_func(output, b_y.long())
            running_loss += float(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss.append(running_loss)
        print("Epoch: " + str(epoch) + "/" + str(EPOCH) + ", loss: " + str(running_loss))
    Loss0 = np.array(Loss)
    np.savetxt('loss_epochs_{0}_rate_{1}.txt'.format(EPOCH, LEARNING_RATE), Loss0)


    torch.save(cnn.state_dict(),"cnn_{0}_{1}.pkl".format(EPOCH, LEARNING_RATE))

if __name__ == "__main__":
    print("train")
    train_model()


