import numpy as np
import pandas as pd
import pydicom
import os
import cv2
import math
import time
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# dataset
import os
import glob

processed_Tdata = []
processed_Rdata = []
img_dimension = 50
num_slices = 20


# simple local helper mean function
def mean(n):
    return (sum(n) / len(n))

# create n-sized chunks from list l
def chunks(l, n):
    count = 0
    for i in range(0, len(l), n):
        if (count < num_slices):
            # fancy python yield statement
            yield l[i:i + n]
            count = count + 1

# function for pre-processing a single patient
def preprocess(patient):
    # reference for reading patient slices: https://www.kaggle.com/dfoozee/data-science-bowl-2017/full-preprocessing-tutorial
    print("read successfully")
    print("\tpatient ID: " + patient)
    path = patient
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))

    chunk_slices = []
    slices = [cv2.resize(np.array(n.pixel_array),(img_dimension,img_dimension)) for n in slices]

    # reference for all of this chunking madness: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    # this code is pretty gross, TODO: clean this bit up
    chunk_sizes = math.floor(len(slices) / num_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        chunk_slices.append(slice_chunk)

    if len(chunk_slices) == num_slices - 1:
        chunk_slices.append(chunk_slices[-1])

    if len(chunk_slices) == num_slices - 2:
        chunk_slices.append(chunk_slices[-1])
        chunk_slices.append(chunk_slices[-1])

    if len(chunk_slices) == num_slices + 2:
        val = list(map(mean, zip(*[chunk_slices[num_slices-1],chunk_slices[num_slices],])))
        del chunk_slices[num_slices]
        chunk_slices[num_slices-1] = val
        
    if len(chunk_slices) == num_slices + 1:
        val = list(map(mean, zip(*[chunk_slices[num_slices-1],chunk_slices[num_slices],])))
        del chunk_slices[num_slices]
        chunk_slices[num_slices-1] = val

    return np.array(chunk_slices)



def kyocera_data(data_path):

    #typedata : numpy array
    #typelabel : [0 0 0 1] #normal_cp #abnormal_cp #normal_smd #abnormal_smd
    x_train_s, x_test_s, x_test_b = [], [], []
    x_ref, y_ref = [], []
    x_test_s_path , x_test_b_path =[] ,[]
    cp1_path = os.path.join(data_path, 'kyocera_CP1')
    smd_path = os.path.join(data_path, 'kyocera_SMD')

    #make reference data
    cp1_normal_path = os.path.join(cp1_path, 'train', 'OK')
    cp1_normal = sorted(glob.glob('{}/*'.format(cp1_normal_path)))
    print("This is : ",cp1_normal_path)
    for cp1 in cp1_normal:
        # print("cp1 is :",cp1)
        cp1 = preprocess(cp1)
        x_train_s.append(cp1)
        x_ref.append(cp1)
        y_ref.append(np.array([0, 1]))
        processed_Rdata.append([cp1, np.array([0,1])])
    # #make test data
    cp1_test_path =  os.path.join(cp1_path, 'test')
    cp1_test_normal = os.path.join(cp1_test_path,'OK')
    cp1_test_abnormal = os.path.join(cp1_test_path,'NG')

    cp1_test_normal_files = sorted(glob.glob('{}/*'.format(cp1_test_normal)))
    for cp1_test_norfile in cp1_test_normal_files : 
        cp1_test_norfile = preprocess(cp1_test_norfile)
        x_test_s_path.append(cp1_test_norfile)
        x_test_s.append(cp1_test_norfile)
        processed_Tdata.append([cp1_test_norfile, np.array([0,1])])

    cp1_test_abnormal_files = sorted(glob.glob('{}/*'.format(cp1_test_abnormal)))
    for cp1_test_abnor in cp1_test_abnormal_files:
        cp1_test_abnor = preprocess(cp1_test_abnor)
        x_test_b_path.append(cp1_test_abnor)
        x_test_b.append(cp1_test_abnor)
        processed_Tdata.append([cp1_test_abnor, np.array([0,1])])
    np.save('processed_Rdata.npy', processed_Rdata)
    np.save('processed_Tdata.npy', processed_Tdata)

    #Debug the data path
    print("cp1_normal_path = ",cp1_normal_path)
    print("cp1_test_path = ",cp1_test_path)
    print("cp1_test_normal = ",cp1_test_normal)
    print("cp1_test_abnormal = ",cp1_test_abnormal)
    
    return x_train_s, x_ref, y_ref, x_test_s, x_test_b, x_test_s_path, x_test_b_path