from PyTorch_CNN import CNN
import torch
import imgdata
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np

EPOCH=20
LEARNING_RATE=0.00001
def load_model():
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn_{0}_{1}.pkl'.format(EPOCH, LEARNING_RATE)))
    return cnn

def test_model(cnn):
    cnn.eval()
    total = 0
    correct = 0
    test_set=imgdata.DefaultTestSet()
    test_loader = Data.DataLoader(dataset=test_set)
    conf_arr = np.zeros((5, 5), dtype=np.int)#create a 5*5 matrix with all values 0
    for index, entity in enumerate(test_loader):
        b_x = Variable(entity['imNorm'])
        b_y = Variable(entity['label']).numpy().squeeze()
        b_y = int(b_y)
        pred_y = cnn.predict(b_x)
        print(pred_y)
        print(b_y)
        total+=1
        if b_y==pred_y:
            correct+=1
        conf_arr[b_y][pred_y] += 1

    print(str(correct/total*100) + '%')
    print(correct)
    print(total)

if __name__ == "__main__":
    test_model(load_model())
