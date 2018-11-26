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
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def plot_curves(output_path, model):

    train_losses = model.history[:,'train_loss']
    valid_losses = model.history[:, 'valid_loss']
    accuracies = model.history[:, 'valid_acc']

    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (20,5))
    ax1.set_title("Training losses per epoch")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy")
    ax1.plot(train_losses)

    ax2.set_title("Validation losses per epoch")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Cross Entropy")
    ax2.plot(valid_losses)


    ax3.set_title("Validation accuracies per epoch")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Average accuracy")
    ax3.plot(accuracies)

    fig.savefig(output_path)

def plot_confusion_matrix(output_path, model, cm, classes):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    fig.suptitle('Confusion matrix')
    cm = cm/cm.sum(axis=1)[: np.newaxis]
    fig.colorbar(ax.matshow(cm,cmap = plt.cm.Greys))

    #Fixing tick labels to add first and last dummy
    ticks = [0]
    ticks.extend(classes)
    ticks.append(0)

    # Set up axes
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticklabels(ticks)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    fig.savefig(output_path)
    




def load_data(train_path, label_path, label_encoder):
    train_data = np.load(train_path ,encoding='latin1')[:,1]
    train_labels = pd.read_csv(label_path)
    labels = np.array(train_labels)[:,1]
    labels = label_encoder.fit_transform(labels)
    train_data = np.array([train_data[i].reshape(1,28,28) for i in range(len(train_data))])

    return train_data, labels


if __name__ == '__main__':
    
    #Load and format data
    label_encoder = LabelEncoder()
    train_data, labels = load_data(train_path="/Data/train_rois4.npy",label_path="/Data/train_labels.csv)", label_encoder)

    #Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False


    #Initialize net
    net = NeuralNetClassifier(module=cnn.KaggleNet().double(), criterion=nn.CrossEntropyLoss, optimizer = optim.Adam, 
                            lr= 0.005, max_epochs = 20, train_split = CVSplit(0.1, random_state = 0)
                            , device = device, iterator_train__batch_size = 128, iterator_valid__batch_size=512)

    #Train the model
    net.fit(train_data,labels)

    #Save the model
    net.save_params(f_params='Model outputs/model_params.pkl', 
            f_history='Model outputs/model_history.pkl', f_optimizer='Model outputs/model_optimizer.pkl')


    #Output visuals
    plot_curves(output_path = "Visuals/learning_curves.png", model =net)

    #Get confusion matrix from validation set predictions
    train, valid = CVSplit(0.1,random_state=0).__call__(train_data,labels)
    Xvalid, yvalid = valid.dataset[valid.indices], labels[valid.indices]
    preds = net.predict(Xvalid)
    cm = confusion_matrix(y_pred= preds,y_true=yvalid)
    classes = label_encoder.inverse_transform(range(31))

    plot_confusion_matrix(output_path = "Visuals/confusion_matrix.png", model = net, cm= cm, classes = classes)

    
