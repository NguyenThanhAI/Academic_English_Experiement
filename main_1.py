import argparse
import numpy as np
import pandas as pd


import tensorflow as tf

from models import ANN, pre_process_data
 

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=".")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="inference")
    parser.add_argument("--load", type=str2bool, default=True)

    args = parser.parse_args()

    return args
 
 
if __name__ == '__main__':
    #csv_path = r"C:\Users\Thanh\Downloads\voice_gender\voice.csv"
    #batch_size = 64
#
    #df = pd.read_csv(csv_path)
    #df['label'] = df['label'].replace({'male':1,'female':0})
#
    #x = df.drop("label", axis=1).to_numpy(dtype=np.float)
    #y = df["label"].values
    #labels = np.zeros(shape=(y.shape[0], 2))
    #labels[np.arange(y.shape[0]), y] = 1
    #x = (x - np.min(x, axis=0, keepdims=True))/(np.max(x, axis=0, keepdims=True) - np.min(x, axis=0, keepdims=True))
#
    #skf = StratifiedKFold(n_splits=5)
    #for train_index, test_index in skf.split(x, y):
    #    train_x, test_x = x[train_index], x[test_index]
    #    train_y, test_y = labels[train_index], labels[test_index]

    args = get_args()
    model_dir = args.model_dir
    mode = args.mode
    load = args.load
    learning_rate = 0.05
    batch_size = 256
    n_iterations = 10000

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    train_x, train_y, test_x, test_y = pre_process_data(train_x=train_images, train_y=train_labels, test_x=test_images, test_y=test_labels)
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    #layers_dims = [32, 32, 32, 32, 2]
    layers_dims = [128, 128, 10]

    ann = ANN(layers_dims, model_dir=model_dir, load=load)
    if mode == "train":

        ann.fit(train_x, train_y, test_x, test_y, learning_rate=learning_rate, batch_size=batch_size, n_iterations=n_iterations)
        print("Train Accuracy:", ann.evaluate(train_x, train_y))
        print("Test Accuracy:", ann.evaluate(test_x, test_y))
        ann.plot_cost()
    elif mode == "inference":

        ann.initialize_parameters()
    prediction_array = ann.predict(test_x).T
    prediction_class = np.argmax(prediction_array, axis=-1)
    #print("Predict array: {}, {}".format(prediction_array, prediction_array.shape))
    print("Predicted class: {}, grountruth class: {}".format(prediction_class, test_labels))
    