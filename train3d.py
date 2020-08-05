from keras.applications import MobileNetV2, VGG16, InceptionResNetV2
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, GlobalAveragePooling2D
from keras import backend as K
from keras.engine.network import Network
from keras.datasets import fashion_mnist

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras.callbacks import TensorBoard

from data3d import kyocera_data
from TDcnn import trainR
from TDcnn import trainT
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
input_shape = (139, 139, 3)
classes = 2
batchsize = 64
feature_out = 1536 # secondary network out for Inception Resnetv2
alpha = 0.5 #for MobileNetV2
lambda_ = 0.1 #for compact loss

def original_loss(y_true, y_pred):
    lc = 1/(classes*batchsize) * batchsize**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]   ) / ((batchsize-1)**2)
    return lc

def train(x_target, x_ref, y_ref, epoch_num):

    print("Model build...")
    trainT()
    trainR()
    

if __name__ == "__main__":
    data_path = './data' 
    X_train_s, X_ref, y_ref, X_test_s, X_test_b, _, _ = kyocera_data(data_path)
    print("preprocess finished")
    train(X_train_s, X_ref, y_ref, 3) # 300

    
