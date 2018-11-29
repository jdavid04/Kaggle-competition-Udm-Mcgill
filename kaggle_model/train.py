"""
Usage:
train.py <trainpath> <labelpath> [-v] [(-s <savedir> <name>)] [(-p <predspath>)]

"""


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
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from docopt import docopt

def load_data(train_path, label_path, label_encoder):
    train_data = np.load(train_path ,encoding='latin1')[:,1]
    train_labels = pd.read_csv(label_path)
    labels = np.array(train_labels)[:,1]
    labels = label_encoder.fit_transform(labels)
    train_data = np.array([train_data[i].reshape(1,28,28) for i in range(len(train_data))])

    return train_data, labels, label_encoder

def main(args):

    #Parse command line arguments
    train_path = args['<trainpath>']
    label_path = args['<labelpath>']
    name = args['<name>']
    savedir = args['<savedir>']
    validate = args['-v']
    save = args['-s']
    keep_predictions = args['-p']
    predictions_path = args['<predspath>']

    
    #Load and format data
    label_encoder = LabelEncoder()
    train_data, labels, encoding = load_data(train_path=train_path,label_path=label_path, label_encoder= label_encoder)

    if save:
        joblib.dump(encoding,savedir+'encoding_'+name+'.pkl')

    if keep_predictions:
        train, valid = CVSplit(0.1,random_state=0).__call__(train_data,labels)
        Xvalid, yvalid = valid.dataset[valid.indices], labels[valid.indices]
    

    #Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False


    #Initialize net
    if validate: 
        net = NeuralNetClassifier(module=cnn.KaggleNet().double(), criterion=nn.CrossEntropyLoss, optimizer = optim.Adam, 
                                lr= 0.0001, max_epochs = 40, train_split = CVSplit(0.1, random_state = 0)
                                , device = device, iterator_train__batch_size = 128, iterator_valid__batch_size=512)
    else:
        net = NeuralNetClassifier(module=cnn.KaggleNet().double(), criterion=nn.CrossEntropyLoss, optimizer = optim.Adam, 
                                lr= 0.0001, max_epochs = 40, train_split = None
                                ,device = device, iterator_train__batch_size = 128, iterator_valid__batch_size=512)

    #Train the model
    net.fit(train_data,labels)

    #Save predictions on validation set, if desired
    if keep_predictions:
        preds = net.predict(Xvalid)
        df = pd.DataFrame({"predictions" : preds, "labels": yvalid})
        df.to_csv(predictions_path)

    #Save the model if desired
    if save:
        with open(savedir+'model_'+name+'.pkl', 'wb') as f:
            pickle.dump(net, f)
        with open(savedir+'model_history_'+name+'.json', 'w') as f:
            net.history.to_file(f)


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
    
