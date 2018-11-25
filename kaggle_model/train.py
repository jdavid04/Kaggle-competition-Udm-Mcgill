import numpy as np
from sklearn.preprocessing import LabelEncoder
from kaggle_model import cnn
import torch
import torch.optim as optim
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
import pickle
import pandas as pd

if __name__ == '__main__':
    #Load and format data

    train_data = np.load("/Data/train_rois4.npy", encoding='latin1')[:,1]
    test_data = np.load("/Data/test_rois4.npy", encoding='latin1')[:,1]
    train_labels = pd.read_csv("/Data/train_labels.csv")
    labels = np.array(train_labels)[:,1]
    labels = LabelEncoder().fit_transform(labels)
    train_data = np.array([train_data[i].reshape(1,28,28) for i in range(len(train_data))])

    #Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False


    #Initialize net
    net = NeuralNetClassifier(module=cnn.KaggleNet().double(), criterion=nn.CrossEntropyLoss, optimizer = optim.Adam, 
                            lr= 0.005, max_epochs = 20, train_split = CVSplit(10, random_state = 0)
                            , device = device, iterator_train__batch_size = 64, iterator_valid__batch_size=512)

    #Train the model
    net.fit(train_data,labels)

    #Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(net, f)