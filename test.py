import multiprocessing
from multiprocessing.managers import *
import pickle
import dill
import tensorflow as tf
import keras
from keras import Model
from keras.layers import *
import keras.backend as K
from keras.utils import plot_model

if __name__ == '__main__':
    agent_list = multiprocessing.Manager().list(range(0))
    a =K.placeholder([None, 2],dtype=tf.float32)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    a=a.eval()
    agent_list.append(a)
    print("finish")
