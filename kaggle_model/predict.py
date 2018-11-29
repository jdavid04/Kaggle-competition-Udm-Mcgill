"""
Usage: 
predict.py <data> <encoder> <model> <output>

"""


import pandas as pd
import numpy as np
from sklearn.externals import joblib
import pickle
from docopt import docopt

def main(args):
    data_path = args['<data>']
    encoder_path = args['<encoder>']
    model_path = args['<model>']
    output_path = args['<output>']


    #Load
    data = np.load(data_path,encoding='latin1')[:,1]
    data = np.array([data[i].reshape(1,28,28) for i in range(len(data))])

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    encoder = joblib.load(encoder_path)

    #Predict
    predictions = encoder.inverse_transform(model.predict(data))

    #Save predictions
    df = pd.DataFrame({"Category" : predictions})
    df.index.name = 'Id'
    df.to_csv(output_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)