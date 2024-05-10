#!/usr/bin/env python
from matplotlib import pyplot as plt
from cmath import nan
import random as rnd
import numpy as np
import collections
import os
import math
import sys
sys.path.append('/Users/wuyuheng/Downloads/FL_RD-main/csh-master')
import time
from utils_quantize import *
#from kmeans import *
from models.cifar10_models import build_model
from datetime import datetime

# tf and keras
import tensorflow as tf
#import pyclustering
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
#from sklearn.cluster import KMeans
#from pyclustering.cluster.kmeans import kmeans
#from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.integrate import quad
from csvec import CSVec
import torch
import math
#------------------------------
# DNN settings
learning_rate = 0.01
epochs = 3

# change seed calculate the average
seeds_for_avg = [5, 57, 85, 12, 29]
rnd.seed(seeds_for_avg[0])
np.random.seed(seeds_for_avg[0])
tf.random.set_seed(seeds_for_avg[0])

#torch.cuda.empty_cache()
batch = 128                 # VGG 16    other 32, original 32(by Henry)
#iterations = 50
number_of_users = 10
fraction = [0.1, 0.143, 0.15, 0.2, 0.25, 0.33, 0.4, 0.5, 1] # NEW(by Henry)
#sparsification_percentage = 60

# Slotted ALOHA settings
transmission_probability = 1 / (number_of_users)
# Try different number of slots in one time frame
number_of_slots = [1, 2, 3, 4, 5, 7, 10]
number_of_timeframes = 15

# Dynamic adjustment of transmission probability settings
target_magnitude = None  # Initialize with a sensible default or based on early observations

# sparse_gradient[0].shape

layers_to_be_compressed=np.array([6,12,18,24,30,36,42])

#compression_type="uniform scalar"
#compression_type="uniform scalar with memory"
#compression_type="k-means"
#compression_type="k-means with memory"
#compression_type="sketch"
#compression_type="weibull"
compression_type = "no compression"
# compression_type = "no compression with float16 conversion"

#------------------------------
def train_validation_split(X_train, Y_train):
    train_length = len(X_train)
    validation_length = int(train_length / 4)
    X_validation = X_train[:validation_length]
    X_train = X_train[validation_length:]
    Y_validation = Y_train[:validation_length]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation

def top_k_sparsificate_model_weights_tf(weights, fraction):
    tmp_list = []
    for el in weights:
        lay_list = el.reshape((-1)).tolist()
        tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list.sort(reverse=True)
    print("total number of parameters:",len(tmp_list))
    #TODO
    # same as weight.reshape.size[0] ? better make it more general
    # write as in 183
    k_th_element = tmp_list[int(fraction*len(tmp_list))-1] # 552874 is the number of parameters of the CNNs!       23608202:Res50   0.0004682019352912903
    new_weights = []
    #new_weight = []
    for el in weights:
        '''
        original_shape = el.shape
        reshaped_el = el.reshape((-1))
        for i in range(len(reshaped_el)):
            if abs(reshaped_el[i]) < k_th_element:
                reshaped_el[i] = 0.0
        new_weights.append(reshaped_el.reshape(original_shape))
        '''
        mask = tf.math.greater_equal(tf.math.abs(el), k_th_element)
        new_w = tf.multiply(el, tf.cast(mask, weights[0]))
        new_weights.append(new_w.numpy())
    '''    
    # 60% test
    num=0 
    for el in new_weights:
        num = num + np.count_nonzero(el)
    print("percentage:", num/len(tmp_list))
    '''
    return new_weights

def pdf_doubleweibull(x, a, m, scale=1):
  return stats.dweibull.pdf(x,a,m,scale)

def update_centers_magnitude_distance_weibull(data, R, iterations_kmeans):
    M = QUANTIZATION_M
    mu = np.mean(data)
    s = np.var(data)
    data_normalized = np.divide(np.subtract(data,mu),np.sqrt(s))
    a, m, b = stats.dweibull.fit(data_normalized)
    print(a,m,b)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_doubleweibull(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    return np.add(np.multiply(thresholds_update,np.sqrt(s)),mu), np.add(np.multiply(centers_update,np.sqrt(s)),mu)

def pdf_gennorm(x, a, m, b):
  return stats.gennorm.pdf(x,a,m,b)

def update_centers_magnitude_distance(data, R, iterations_kmeans):
    #TODO: allow change of m
    M = QUANTIZATION_M
    mu = np.mean(data)
    s = np.var(data)
    data_normalized = np.divide(np.subtract(data,mu),np.sqrt(s))
    a, m, b = stats.gennorm.fit(data_normalized)
    print(a,m,b)

    xmin, xmax = min(data_normalized), max(data_normalized)
    random_array = np.random.uniform(0, min(abs(xmin), abs(xmax)), 2 ** (R - 1))
    centers_init = np.concatenate((-random_array, random_array))
    thresholds_init = np.zeros(len(centers_init) - 1)
    for i in range(len(centers_init) - 1):
        thresholds_init[i] = 0.5 * (centers_init[i] + centers_init[i + 1])

    centers_update = np.copy(np.sort(centers_init))
    thresholds_update = np.copy(np.sort(thresholds_init))
    for i in range(iterations_kmeans):
        integ_nom = quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        integ_denom = quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), -np.inf, thresholds_update[0])[0]
        #centers_update[0] = np.divide(integ_nom, integ_denom)
        centers_update[0] = np.divide(integ_nom, (integ_denom + 1e-7))
        for j in range(len(centers_init) - 2):          # j=7
            integ_nom_update = \
            quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            integ_denom_update = \
            quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), thresholds_update[j], thresholds_update[j + 1])[0]
            ###
            centers_update[j + 1] = np.divide(integ_nom_update, (integ_denom_update + 1e-7))
            #if (np.abs(integ_nom_update)<0.0000000001) or (np.abs(integ_denom_update)<0.0000000001):
            #    centers_update[j + 1] = 0
            #else:
            #    centers_update[j + 1] = np.divide(integ_nom_update, integ_denom_update)  # integ_denom_update+eplison
        integ_nom_final = \
        quad(lambda x: x ** (M+1) * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        integ_denom_final = \
        quad(lambda x: x ** M * pdf_gennorm(x, a, m, b), thresholds_update[len(thresholds_update) - 1], np.inf)[0]
        #centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, integ_denom_final)
        centers_update[len(centers_update) - 1] = np.divide(integ_nom_final, (integ_denom_final+ 1e-7))
        for j in range(len(thresholds_update)):
            thresholds_update[j] = 0.5 * (centers_update[j] + centers_update[j + 1])
    #thresholds_final = np.divide(np.subtract(thresholds_update,thresholds_update[::-1]),2)
    #centers_final = np.divide(np.subtract(centers_update,centers_update[::-1]),2)
    return np.add(np.multiply(thresholds_update,np.sqrt(s)),mu), np.add(np.multiply(centers_update,np.sqrt(s)),mu)

def fp8_152_bin_edges(exponent_bias=15):
    bin_centers = np.zeros(247,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(247, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(31):
            for f in range(4):
                if e != 0:
                    exponent = e - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[bin_centers[idx]] = str(s) + format(e, 'b').zfill(5) + format(f, 'b').zfill(2)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[bin_centers[idx]] = str(s) + format(e, 'b').zfill(5) + format(f,'b').zfill(2)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict[0.0] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    #print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict

def simulate_transmissions(number_of_users, transmission_probability):
    """
    Simulate the transmission decision of each user.
    
    Args:
    - number_of_users (int): The total number of users participating.
    - transmission_probability (float): The probability of a user deciding to transmit.
    
    Returns:
    - successful_users (list): A list of users who successfully transmitted without collision.
    """
    # 動態改變種子
    current_time = int(time.time())
    np.random.seed(current_time % 123456789)
    
    # Each user decides to transmit based on the transmission probability
    decisions = np.random.rand(number_of_users) < transmission_probability
    
    # Identify successful transmissions (exactly one transmission)
    if np.sum(decisions) == 1:
        successful_users = [i for i, decision in enumerate(decisions) if decision]
    else:
        successful_users = []  # Collision or no transmission, no successful user
    
    return successful_users

def calculate_update_memory_magnitude(temp_memory_matrix):
    if not temp_memory_matrix:  # Check if the list is empty
        return 0  # Return 0 or another sensible default if no data to calculate magnitude
    magnitudes = [np.linalg.norm(memory) for memory in temp_memory_matrix]
    if magnitudes:  # Further check to ensure magnitudes list is not empty
        return np.mean(magnitudes)
    return 0

def adjust_transmission_probability(current_prob, target_magnitude, current_magnitude, number_of_users, min_prob=0.1, max_prob=1.0, sensitivity=0.05):
    if target_magnitude == 0 and current_magnitude == 0:
        if current_prob > 0.2:
            new_prob = 1 / number_of_users
        else:
            new_prob = current_prob + sensitivity / 2
    else:
        error = (target_magnitude - current_magnitude) / (target_magnitude + 1e-7)
        adjustment = sensitivity * error
        new_prob = np.clip(current_prob + adjustment, min_prob, max_prob)
    return new_prob

def print_model_size(mdl):
    #torch.save(mdl.state_dict(), "tmp.pt")
    mdl.save_weights('./checkpoints/tmp')
    size = os.path.getsize('./checkpoints/tmp.data-00000-of-00001')
    print("%.2f MB" %(size /1e6))
    os.remove('./checkpoints/tmp.data-00000-of-00001')

classes = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck",
}

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
num_classes = len(classes)

# normalize to one
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# categorical loss enropy
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

# About Model
# Load the VGG-16 model pre-initialized weights
model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Adding custom layers on top of VGG-16
model = tf.keras.Sequential([
    model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

if model._name == 'VGG16':
    opt = Adam(learning_rate=0.00005)
    number_threshold = 500000
elif model._name == 'resnet18':
    opt = Adam()
    number_threshold = 100000
else:
    # DNN
    opt = Adam(learning_rate=0.0001)
    number_threshold = 1000

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

num_comp = 0
large_layers = []
for i in range(len(model.get_weights())):
    if model.get_weights()[i].size > number_threshold:
        large_layers.append(i)
        num_comp = num_comp + model.get_weights()[i].size
layers_to_be_compressed = np.asarray(large_layers)
print("layers to be compressed:", layers_to_be_compressed)
print("Compressing number:", num_comp)    
# layers_to_be_compressed=np.array([6,12,18,24,30,36,42])   DNN
# layers to be compressed: [ 72  78  96 114 132 144 150 156 158 168 174 180 186 192 198 204 210 216 222 228 234 240 246 252 258 264 270 272 282 288 294 300 306 312]

# FL setting
size_of_user_ds = int(len(X_train)/number_of_users)
train_data_X = np.zeros((number_of_users,size_of_user_ds, 32, 32, 3))
train_data_Y = np.ones((number_of_users,size_of_user_ds,10))
for i in range(number_of_users):
    train_data_X[i] = X_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    train_data_Y[i] = Y_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]    

import argparse
parser = argparse.ArgumentParser(description='Federared Learning Parameter')
parser.add_argument('--R', type=int, default=1,
                    help='How many bit rate')
parser.add_argument('--M', type=int, default=0,
                    help='M for K-means Quantizer')
parser.add_argument('--P', type=int, default=60,
                    help='Sparsification percentage')

args = parser.parse_args()

# Parameter
QUANTIZATION_M = args.M
BIT_RATE = args.R
sparsification_percentage = args.P

d = 1
# indexed by dimenions of the input, so far we go through each layer separately
rate = np.array([d])
rate[0] = BIT_RATE

# # this is an array too, 
# # indexed as [rate][memory]
# c_scale = np.ones([10,10])

# -----------------------------------------------------------------------------
iter = 0

w_before_train = model.get_weights()

from datetime import datetime
timeObj = datetime.now().time()
timeStr = timeObj.strftime("%H_%M_%S_%f")
out_file = "accuracy-"+compression_type+"-R"+str(BIT_RATE)+"-M" +str(QUANTIZATION_M)+"-fixed-seed9"

print(compression_type)
print("M = ", QUANTIZATION_M)
print("Rate = ", rate[0])
gradient_list = []

index_for_plot = 2
user_metrics = {'loss': {i + 1: [] for i in range(number_of_users)},
                'accuracy': {i + 1: [] for i in range(number_of_users)}}
global_model_accuracies = []  # Store the global model's accuracies over iterations

# This is for plotting
accuracies_for_each_iter = {user_id: [] for user_id in range(1, number_of_users + 1)}
success_user = None

# This is momentum for memory matrix
gamma_momentum = [1, 0.9, 0.8, 0.7, 0.5, 0.1]

with open(out_file + timeStr + '.txt', "w") as outfile:
    memory_matrix = [[np.zeros_like(weight) for weight in w_before_train] for _ in range(number_of_users)]
    iteration_start_time = datetime.now()  # Capture start time of the iteration
    temp_memory_matrix = []
    
    for _ in range(number_of_timeframes):
      print('')
      print("************ Timeframe " + str(_ + 1) + " ************")
      _, accuracy = model.evaluate(X_test, Y_test)
      print('Test accuracy BEFORE this time frame is', accuracy)  
    
      sum_terms = []
      # Send the updated weights from server to all users
      wc = model.get_weights()
    
      for slot in range(number_of_slots[4]):
          print()
          print("**** Slot " + str(slot + 1) + " ****")
          iter = iter + 1

          #beta = Beta[iter-1]
          beta = 0.3
          #print("beta: ", beta)
      
          # Simulate transmissions for this slot, see which user successfully transmitted
          successful_users = simulate_transmissions(number_of_users, transmission_probability)
          if successful_users:  # Check if the list is not empty
            success_user = successful_users[0] + 1

          for i in range(number_of_users):
            if i in successful_users:
                # Update the local model with the latest weights from the server before training
                # Ensure local model starts with the latest global model weights
                model.set_weights(wc)
            
                X_train_u = train_data_X[i]
                Y_train_u = train_data_Y[i]
                np.random.seed(5)
                shuffler = np.random.permutation(len(X_train_u))
                X_train_u = X_train_u[shuffler]
                Y_train_u = Y_train_u[shuffler]
                X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)

                print()
                print('user->', i + 1)
                print("beta: ", beta)
                #print(len(X_train_u))
                history = model.fit(x=X_train_u,y=Y_train_u,
                                      epochs = epochs,
                                      batch_size = batch,
                                      validation_data=(X_validation_u, Y_validation_u),
                                      shuffle=False
                                      )

                # Extend the lists with metrics from all epochs
                user_metrics['loss'][i + 1].append(history.history['loss'][-1])  # Append all epoch losses
                user_metrics['accuracy'][i + 1].append(history.history['accuracy'][-1])  # Append all epoch accuracies
                accuracies_for_each_iter[i + 1].append(history.history['accuracy'][-1])
                # check model size
                #print_model_size(model)
                # compare model

                _, accuracy = model.evaluate(X_test, Y_test)

                # TODO
                # i'd use a more meaningful name ^_^
                # Communication PS->clients
                wu = model.get_weights()

                nu = len(Y_train_u)+len(Y_validation_u)
                frac = nu/len(Y_train)

                # approx gradient with model difference
                gradient = [np.subtract(wu[k], wc[k]) for k in range(len(wu))]
                gradient_with_memory = [gradient[j] + memory_matrix[i][j] for j in range(len(gradient))]
                #print('sparse level:', sparsification_percentage/100)
                #sparse_gradient = top_k_sparsificate_model_weights_tf(gradient, sparsification_percentage/100)
                #print('sparse level:', sparsification_percentage/(BIT_RATE*100))
                print('sparse level:', fraction[3]) # NEW(by Henry)
                #sparse_gradient = top_k_sparsificate_model_weights_tf(gradient, sparsification_percentage/(BIT_RATE*100))
                sparse_gradient = top_k_sparsificate_model_weights_tf(gradient_with_memory, fraction[3]) # NEW(by Henry)
            
                for j in range(len(wc)):
                    memory_matrix[i][j] = gamma_momentum[0] * memory_matrix[i][j] + gradient_with_memory[j] - sparse_gradient[j]
                    temp_memory_matrix.append(memory_matrix[i][j])
                
                # uncomment this line to try the original gradient instead of the sparsed one(by Henry)
                #sparse_gradient = gradient

                #for j in range(len(sparse_gradient)):
                layer_index = 1
                for j in layers_to_be_compressed:
              # np.savetxt(outfile, [np.reshape(gradient[j],(np.size(gradient[j],))), ], fmt='%10.3e', delimiter=',')
              # the size of this is 44
              # I would skip all the layers that have a small size.
              # only compress the ones in layers_to_be_compressed
                  gradient_shape = np.shape(sparse_gradient[j])
                  gradient_size = np.size(sparse_gradient[j])
                  gradient_reshape = np.reshape(sparse_gradient[j],(gradient_size,))
                  non_zero_indices = tf.where(gradient_reshape != 0).numpy()

          # reshaping the memory
          # memory_shape = np.shape(memory_array[iter-1,i,j])
          # memory_size = np.size(memory_array[iter-1,i,j])
          # memory_reshape = np.reshape(memory_array[iter-1,i,j],(memory_size,))

          # Modified version
          # First, access the correct array for the current iteration
          # Not necessary to print if no compression is done(by Henry)
          #print("Layer",j,": entries to compress:",non_zero_indices.size, "total # entries:", gradient_size )


                  if compression_type == "sketch":
                        sparse_g_tensor = torch.tensor(sparse_gradient[j]).to(device='cuda')
                        sparse_g_tensor_flatten = sparse_g_tensor.view(-1)
                # perform sketch to this seq
                # should we perform sketch on weight tensors or on non-zero indices?
                        num_rows = 5
                        num_cols = math.floor(gradient_size/(1*num_rows))        # Change to change the rate?   10
                        sketch = CSVec(d=gradient_size, c=num_cols, r=num_rows)   # device="cpu", numBlocks=1
                        sketch.accumulateVec(sparse_g_tensor_flatten)
                        seq_enc = sketch.table

                # unsketch
                #num_nonzero = gradient_size
                        num_nonzero = len(non_zero_indices)
                        sketch = CSVec(d=gradient_size, c=num_cols, r=num_rows)
                        sketch.accumulateTable(seq_enc)
                        seq_dec = sketch.unSketch(k=num_nonzero)  
                # compare  seq_dec with sparse_g_tensor_flatten
                # assert match_shape
                        sparse_gradient[j] = torch.reshape(seq_dec, gradient_shape).cpu().numpy()
                        continue


          #SR2SS
          # i would say > 1000, no need to worry about the small dimensions here
                  if (non_zero_indices.size > 1):
                      seq = gradient_reshape[np.transpose(non_zero_indices)[0]]
              #mem_seq = memory_reshape[np.transpose(non_zero_indices)[0]]


                      if  compression_type=="uniform scalar":

                          seq_enc, uni_max, uni_min= compress_uni_scalar(seq, rate)


                          seq_dec = decompress_uni_scalar(seq_enc, rate, uni_max, uni_min)

                      elif  compression_type=="gaussian scalar":
                         seq_enc, mu, s = gaussian_compress(seq, rate[0])
                         seq_dec = decompress_gaussian(seq_enc, mu, s)

                      elif compression_type=="k-means":
                              thresholds, quantization_centers = update_centers_magnitude_distance(data=seq, R=rate[0],  iterations_kmeans=100)
                              thresholds_sorted = np.sort(thresholds)
                              labels = np.digitize(seq,thresholds_sorted)
                              index_labels_false = np.where(labels == 2**rate[0])
                              labels[index_labels_false] = 2**rate[0]-1
                              seq_dec = quantization_centers[labels]

                      elif compression_type=="weibull":
                              thresholds, quantization_centers = update_centers_magnitude_distance_weibull(data=seq, R=rate[0],  iterations_kmeans=100)
                              thresholds_sorted = np.sort(thresholds)
                              labels = np.digitize(seq,thresholds_sorted)
                              index_labels_false = np.where(labels == 2**rate[0])
                              labels[index_labels_false] = 2**rate[0]-1
                              seq_dec = quantization_centers[labels]

                      elif compression_type == "k-means with memory":
                      #beta = 0.5
                          seq_to_be_compressed = seq+beta*mem_seq
                          thresholds, quantization_centers = update_centers_magnitude_distance(data=seq_to_be_compressed, R=rate[0],  iterations_kmeans=100)
                          thresholds_sorted = np.sort(thresholds)
                          labels = np.digitize(seq, thresholds_sorted)
                          index_labels_false = np.where(labels == 2 ** rate[0])
                          labels[index_labels_false] = 2 ** rate[0] - 1
                          seq_dec = quantization_centers[labels]
                          seq_error = beta*mem_seq+seq_to_be_compressed-seq_dec
                          np.put(memory_reshape, np.transpose(non_zero_indices)[0], seq_error)

                          memory_array[iter,i,j] = memory_reshape.reshape(memory_shape)
                  # SR2SS
                  # need to stare overything: see how it changes over layer and over time

                      elif compression_type == "optimal compression":
                          seq_enc , mu, s = optimal_compress(seq,rate)
                          seq_dec = decompress_gaussian(seq_enc, mu, s)

                      elif compression_type == "no compression":
                          seq_dec = seq

                      elif compression_type == "no compression with float16 conversion":
                          seq_dec = seq.astype(np.float16)

                      elif compression_type == "no compression with float8 conversion":
                          fp8_bin_centers, fp8_bin_edges, fp8_dict = fp8_152_bin_edges()
                          indices = np.digitize(seq, fp8_bin_edges)
                          seq_dec = fp8_bin_centers[indices]


              # compress_decompress(type='TCQ')

              #plot the histogram of data

              #saving the histogram after compression
              #dec_max = np.amax(seq_dec)
              #dec_min = np.amin(seq_dec)
              #step_size = (dec_max - dec_min) / 100
              #bins_array_dec = np.arange(dec_min, dec_max, step_size)
              #hist_after, bin_edges_after = np.histogram(seq_dec, bins=bins_array_dec)


              #saving histogram after compression
              #if ((j== 6) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[1],header='#layer6-after comp-histogram')
              #    for bin_index in range(len(bin_edges_after)-1):
              #        np.savetxt(outfile, [[bin_edges_after[bin_index],hist_after[bin_index]],],fmt='%10.3e', delimiter=',')
              #if ((j== 24) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[2],header='#layer24-after comp-histogram')
              #    for bin_index in range(len(bin_edges_after)-1):
              #        np.savetxt(outfile, [[bin_edges_after[bin_index],hist_after[bin_index]],],fmt='%10.3e',delimiter =',')
              #if ((j == 42) & (i == 0) & (iter == 10)):
              #    np.savetxt(outfile, [3], header='#layer42-after comp-histogram')
              #    for bin_index in range(len(bin_edges_after) - 1):
              #        np.savetxt(outfile, [[bin_edges_after[bin_index], hist_after[bin_index]],], fmt='%10.3e',delimiter=',')

              #unique_labels, unique_indices, counts = np.unique(seq_dec,return_index=True,return_counts=True)
              #if ((j== 12) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[1],header='#layer12-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],],fmt='%10.3e',delimiter=',')
              #if ((j== 24) & (i==0) & (iter==10)):
              #    np.savetxt(outfile,[2],header='#layer24-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],],fmt='%10.3e', delimiter =',')
              #if ((j == 42) & (i == 0) & (iter == 10)):
              #    np.savetxt(outfile, [3], header='#layer42-after comp-unique')
              #    for bin_index in range(len(unique_labels)):
              #        np.savetxt(outfile, [[unique_labels[bin_index],counts[bin_index]],], fmt='%10.3e',delimiter=',')
              #np.savetxt(outfile, [bin_edges_after])
              #np.savetxt(outfile, [hist_after])
              #fig = plt.figure()
              #ax = fig.add_subplot(1, 1, 1)
              #ax.hist(seq_dec, bins=bins_array)
              #plt.xlabel('bins')
              #plt.ylabel('histogram of quantized data')
              #fig.savefig('hist-after compression-'+'Iter'+str(iter)+'-Layer'+ str(j)+'.png')

                      np.put(gradient_reshape, np.transpose(non_zero_indices)[0], seq_dec)

                      sparse_gradient[j] = gradient_reshape.reshape(gradient_shape)
                      layer_index = layer_index + 1

        #user_gradient = [np.add(wc[i], sparse_gradient[i]) for i in range(len(sparse_gradient))]
        #gradient_list.append(user_gradient)
        
        # this is the PS part
        # Communication clients to PS
        # The result is a list of weighted gradients for each layer of the model from one client.
        # These weighted gradients are appended to the 'sum_terms' list, which collects such contributions from all clients.
                sum_terms.append([np.multiply(frac, grad) for grad in sparse_gradient])
            else:
                print(f"User {i + 1} did not transmit or collision occurred.")
            # Resetting model weights to their pre-update state, which is wc
            model.set_weights(wc)

          # 原本畫圖的位置
          # Plotting
          fig, axs = plt.subplots(2, 1, figsize=(10, 8))     
          # Plot for Loss
          for user, losses in user_metrics['loss'].items():
              axs[0].plot(range(1, len(losses) + 1), losses, label=f'User {user}', marker='o')
          axs[0].set_title('Loss per User over Slots')
          axs[0].set_xlabel('Slot')
          axs[0].set_ylabel('Loss')
          axs[0].legend(loc = 'upper left', bbox_to_anchor=(1, 1))
      
          # Plot for Accuracy
          for user, accuracies in user_metrics['accuracy'].items():
              axs[1].plot(range(1, len(accuracies) + 1), accuracies, label=f'User {user}', marker='o')
          axs[1].set_title('Accuracy per User over Slots')
          axs[1].set_xlabel('Slot')
          axs[1].set_ylabel('Accuracy')
          axs[1].legend(loc = 'upper left', bbox_to_anchor=(1, 1))
          
          plt.tight_layout()
          plt.show()

      if sum_terms:
        update = sum_terms[0]
        for i in range(1, len(sum_terms)):
            tmp = sum_terms[i]
            update = [np.add(tmp[j], update[j]) for j in range(len(update))]
            # After going through all users, "update" contains the combined updates from all users.
            # It's like gathering bits of knowledge from everyone and putting it all together.
        new_weights = [np.add(wc[i], update[i]) for i in range(len(wc))]
        model.set_weights(new_weights)
      else:
        print("No successful transmissions; skipping update.")
       
      # check test accuracy
      results = model.evaluate(X_test, Y_test)
      global_model_accuracies.append(results[1])
      
      # Update the accuracies for plotting
      for key in accuracies_for_each_iter:
        if key != success_user or success_user is None:
            accuracies_for_each_iter[key].append(results[1])
      success_user = None
      
      # check the performance at the PS, monitor the noise
      print()
      print('Test accuracy AFTER PS aggregation',results[1])

      plt.figure(figsize=(10, 4))
      plt.plot(range(1, index_for_plot), global_model_accuracies, label='Global Accuracy', marker = 'o')
      plt.title('Global Model Accuracy over Time Frames')
      plt.xlabel('Timeframe')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.show()
      
      index_for_plot = index_for_plot + 1
      
      # Dynamically adjust the transmission probability
      current_magnitude = calculate_update_memory_magnitude(temp_memory_matrix)
      print("Current Magnitude: "+ str(current_magnitude))
      if target_magnitude is None:
        target_magnitude = current_magnitude  # Set initial target based on first timeframe
      print("Target Magnitude: " + str(target_magnitude))
      transmission_probability = adjust_transmission_probability(transmission_probability,
                                                                 target_magnitude,
                                                                 current_magnitude,
                                                                 number_of_users
                                                                )
      print("New transmission probability: " + str(transmission_probability))
      
      target_magnitude = current_magnitude
      temp_memory_matrix = []
      #np.savetxt(outfile, [[int(iter),results[1]],],fmt='%10.3e',delimiter =',')
      #np.savetxt(outfile, [results[1]],fmt='%10.3e')
    iteration_end_time = datetime.now()  
    iteration_duration = iteration_end_time - iteration_start_time
    print(f'This total process took {iteration_duration} to complete.')