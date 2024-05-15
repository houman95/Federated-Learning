#!/usr/bin/env python
from matplotlib import pyplot as plt
import random as rnd
import numpy as np
import sys
sys.path.append('/Users/wuyuheng/Downloads/FL_RD-main/csh-master')
import time
from datetime import datetime

# tf and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications import VGG16
#------------------------------
# DNN settings
learning_rate = 0.01
epochs = 3

# change seed calculate the average
seeds_for_avg = [5, 57, 85, 12, 29]
rnd.seed(seeds_for_avg[4])
np.random.seed(seeds_for_avg[4])
tf.random.set_seed(seeds_for_avg[4])

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
        if current_prob > 0.125:
            new_prob = 1 / number_of_users
        else:
            new_prob = current_prob + sensitivity / 2
    else:
        error = (target_magnitude - current_magnitude) / (target_magnitude + 1e-7)
        adjustment = sensitivity * error
        new_prob = np.clip(current_prob + adjustment, min_prob, max_prob)
    return new_prob

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
                # uncomment this line to try the original gradient instead of the sparsed one(by Henry)
                #sparse_gradient = gradient
            
                for j in range(len(wc)):
                    memory_matrix[i][j] = gamma_momentum[0] * memory_matrix[i][j] + gradient_with_memory[j] - sparse_gradient[j]
                    temp_memory_matrix.append(memory_matrix[i][j])               
        
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
        
        if num_active_users > 0:
            update = [np.divide(u, num_active_users) for u in update]
            
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
      
    iteration_end_time = datetime.now()  
    iteration_duration = iteration_end_time - iteration_start_time
    print(f'This total process took {iteration_duration} to complete.')