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
rnd.seed(seeds_for_avg[0])
np.random.seed(seeds_for_avg[0])
tf.random.set_seed(seeds_for_avg[0])

#torch.cuda.empty_cache()
batch = 128                 # VGG 16    other 32, original 32(by Henry)
#iterations = 50
number_of_users = 10
fraction = [0.1, 0.15, 0.2] # NEW(by Henry)

# Slotted ALOHA settings
transmission_probability = 1 / (number_of_users)
# Try different number of slots in one time frame
number_of_slots = [3, 5, 10]
number_of_timeframes = 15

compression_type = "no compression"

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
    k_th_element = tmp_list[int(fraction*len(tmp_list))-1]
    new_weights = []
    for el in weights:
        mask = tf.math.greater_equal(tf.math.abs(el), k_th_element)
        new_w = tf.multiply(el, tf.cast(mask, weights[0]))
        new_weights.append(new_w.numpy())
    return new_weights

def simulate_transmissions(number_of_users, transmission_probability):
    current_time = int(time.time())
    np.random.seed(current_time % 123456789)
    
    decisions = np.random.rand(number_of_users) < transmission_probability
    
    if np.sum(decisions) == 1:
        successful_users = [i for i, decision in enumerate(decisions) if decision]
    else:
        successful_users = []  
    
    return successful_users

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

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

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
    opt = Adam(learning_rate=0.0001)
    number_threshold = 1000

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

size_of_user_ds = int(len(X_train)/number_of_users)
train_data_X = np.zeros((number_of_users,size_of_user_ds, 32, 32, 3))
train_data_Y = np.ones((number_of_users,size_of_user_ds,10))
for i in range(number_of_users):
    train_data_X[i] = X_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    train_data_Y[i] = Y_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]    

iter = 0

w_before_train = model.get_weights()

print(compression_type)
gradient_list = []

index_for_plot = 2
user_metrics = {'loss': {i + 1: [] for i in range(number_of_users)},
                'accuracy': {i + 1: [] for i in range(number_of_users)}}
global_model_accuracies = []

accuracies_for_each_iter = {user_id: [] for user_id in range(1, number_of_users + 1)}
success_user = None

gamma_momentum = [1, 0.9, 0.8, 0.7, 0.5, 0.1]

epochs_range = range(1, 11)
num_active_users_range = range(1, 11)
num_channel_sims = 10

memory_matrix = [[np.zeros_like(weight) for weight in w_before_train] for _ in range(number_of_users)]
iteration_start_time = datetime.now()
for _ in range(number_of_timeframes):
    print('')
    print("************ Timeframe " + str(_ + 1) + " ************")
    _, accuracy = model.evaluate(X_test, Y_test)
    print('Test accuracy BEFORE this time frame is', accuracy)  

    sum_terms = []
    num_active_users = 0
    wc = model.get_weights()

    for slot in range(number_of_slots[0]):
        print()
        print("**** Slot " + str(slot + 1) + " ****")
        iter = iter + 1

        successful_users = simulate_transmissions(number_of_users, transmission_probability)
        if successful_users:
            success_user = successful_users[0] + 1

        for i in range(number_of_users):
            if i in successful_users:
                num_active_users = num_active_users + 1
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
                history = model.fit(x=X_train_u,y=Y_train_u,
                                      epochs = epochs,
                                      batch_size = batch,
                                      validation_data=(X_validation_u, Y_validation_u),
                                      shuffle=False
                                      )

                user_metrics['loss'][i + 1].append(history.history['loss'][-1])
                user_metrics['accuracy'][i + 1].append(history.history['accuracy'][-1])
                accuracies_for_each_iter[i + 1].append(history.history['accuracy'][-1])

                _, accuracy = model.evaluate(X_test, Y_test)

                wu = model.get_weights()

                nu = len(Y_train_u)+len(Y_validation_u)
                frac = nu/len(Y_train)

                gradient = [np.subtract(wu[k], wc[k]) for k in range(len(wu))]
                gradient_with_memory = [gradient[j] + memory_matrix[i][j] for j in range(len(gradient))]
                print('sparse level:', fraction[2])
                sparse_gradient = top_k_sparsificate_model_weights_tf(gradient_with_memory, fraction[2])
            
                for j in range(len(wc)):
                    memory_matrix[i][j] = gamma_momentum[0] * memory_matrix[i][j] + gradient_with_memory[j] - sparse_gradient[j]
       
                sum_terms.append([np.multiply(frac, grad) for grad in sparse_gradient])
            else:
                print(f"User {i + 1} did not transmit or collision occurred.")
            model.set_weights(wc)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))     
        for user, losses in user_metrics['loss'].items():
            axs[0].plot(range(1, len(losses) + 1), losses, label=f'User {user}', marker='o')
        axs[0].set_title('Loss per User over Slots')
        axs[0].set_xlabel('Slot')
        axs[0].set_ylabel('Loss')
        axs[0].legend(loc = 'upper left', bbox_to_anchor=(1, 1))
  
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
        
        if num_active_users > 0:
            update = [np.divide(u, num_active_users) for u in update]
        
        new_weights = [np.add(wc[i], update[i]) for i in range(len(wc))]
        model.set_weights(new_weights)
    else:
        print("No successful transmissions; skipping update.")

    results = model.evaluate(X_test, Y_test)
    global_model_accuracies.append(results[1])
    
    for key in accuracies_for_each_iter:
        if key != success_user or success_user is None:
            accuracies_for_each_iter[key].append(results[1])
    success_user = None
    
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

iteration_end_time = datetime.now()  
iteration_duration = iteration_end_time - iteration_start_time
print(f'This total process took {iteration_duration} to complete.')
