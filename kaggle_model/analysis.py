"""
This module uses a model's training history and prediction on a test/validation set to
generate learning curves, a confusion matrix and a performance report. Input/outputs 
are determined by the command line arguments as expressed below.

Usage: 
analysis.py <histpath> <encpath> <predspath> <outdir> <id>
"""
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.ticker as ticker
from skorch.history import History
from docopt import docopt
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def plot_curves(output_path, history):

    train_losses = history[:,'train_loss']
    valid_losses = history[:, 'valid_loss']
    accuracies = history[:, 'valid_acc']

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

def plot_confusion_matrix(output_path, cm, classes):
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    fig.suptitle('Confusion matrix')
    cm = cm/cm.sum(axis=1)[: np.newaxis]
    fig.colorbar(ax.matshow(cm,cmap = plt.cm.Greens))

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

def evaluate_performance(output_path, preds, labels, classes):
    
    report = precision_recall_fscore_support(labels, preds)
    #Compute class accuracies
    class_acc = []
    for i in range(31):
        idx = np.where(labels == i)
        class_acc.append(np.mean(preds[idx] == i).round(2))
        
    out_dict = {
                "precision" : report[0].round(2)
                ,"recall" : report[1].round(2)
                ,"f1-score" : report[2].round(2)
                , "accuracy" : class_acc
                ,"support" : report[3]
                }
    out_df = pd.DataFrame(out_dict, index = classes)


        
    #Compute average
    averages = out_df.mean(axis = 0).round(2).to_frame().T
    averages.index = ["average"]

    #Compute weighted average
    w_precision = (np.array(out_df.precision * out_df.support).sum()/(out_df.support.sum())).round(2)
    w_recall = (np.array(out_df.recall * out_df.support).sum()/(out_df.support.sum())).round(2)
    w_f1 = (np.array(out_df['f1-score'] * out_df.support).sum()/(out_df.support.sum())).round(2)
    w_accuracy = (np.array(out_df.accuracy * out_df.support).sum()/(out_df.support.sum())).round(2)
    w_support = np.mean(out_df.support).round(2)
    w_averages = pd.DataFrame({'precision' : w_precision,'recall': w_recall, 
                            'f1-score': w_f1,'accuracy' : w_accuracy,'support': w_support},index=['weighted average'])
    w_averages.index = ['weighted average']


    out_df = out_df.append(averages)
    out_df = out_df.append(w_averages)


    out_df.to_csv(output_path)

def main(args):
    #Parse args
    outdir  = args['<outdir>']
    id = args['<id>']
    history_path = args['<histpath>']
    predictions_path = args['<predspath>']
    encoding_path = args['<encpath>']

    #Load model    
    with open(history_path, 'r') as f:
        history = History.from_file(f)
    encoding = joblib.load(encoding_path)
    pred_df = pd.read_csv(predictions_path)
    preds = pred_df['predictions']
    labels = pred_df['labels']

    #Output visuals
    plot_curves(output_path = outdir+"learning_curves_"+id+".png", history = history)

    #Get confusion matrix from validation set predictions
    
    cm = confusion_matrix(y_pred= preds,y_true=labels)
    classes = encoding.inverse_transform(range(31))

    plot_confusion_matrix(output_path = outdir+"confusion_matrix_"+id+".png", cm= cm, classes = classes)
    evaluate_performance(output_path=outdir+"performance_"+id+".csv", preds = np.array(preds),
                        labels = np.array(labels), classes = classes)

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)