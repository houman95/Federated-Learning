#!/usr/bin/env python
from matplotlib import pyplot as plt
import random as rnd
import numpy as np
import time
from datetime import datetime
import pandas as pd
# tf and keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications import VGG16
import argparse
import sys

# Simulate command-line arguments
sys.argv = [
    'placeholder_script_name', 
    '--learning_rate', '0.0001',
    '--epochs', '3',
    '--batch_size', '64',
    '--num_users', '10',
    '--fraction', '0.2',
    '--transmission_probability', '0.1',
    '--num_slots', '10',
    '--num_timeframes', '15',
    '--seeds', '42', '57', '85', '12', '29', '33', '7', '91',
    '--gamma_momentum', '1',
    '--num_channel_sims', '100',
    '--use_memory_matrix', 'true'
]

# Define command-line arguments
parser = argparse.ArgumentParser(description="Federated Learning with Slotted ALOHA and CIFAR-10 Dataset")

# Hyperparameters
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--num_users', type=int, default=10, help='Number of users in federated learning')
parser.add_argument('--fraction', type=float, nargs='+', default=[0.1, 0.15, 0.2, 0.4], help='Fractions for top-k sparsification')

# Slotted ALOHA settings
parser.add_argument('--transmission_probability', type=float, default=0.1, help='Transmission probability for Slotted ALOHA')
parser.add_argument('--num_slots', type=int, nargs='+', default=[5, 10, 20], help='Number of slots for Slotted ALOHA simulation')
parser.add_argument('--num_timeframes', type=int, default=15, help='Number of timeframes for simulation')

# Other settings
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 57, 85, 12, 29, 33, 7, 91], help='Random seeds for averaging results')
parser.add_argument('--gamma_momentum', type=float, nargs='+', default=[1, 0.9, 0.8, 0.7, 0.5, 0.1], help='Momentum for memory matrix')
parser.add_argument('--num_channel_sims', type=int, default=5, help='Number of channel simulations')
parser.add_argument('--use_memory_matrix', type=str, default='true', help='Switch to use memory matrix (true/false)')

# Parse arguments
args = parser.parse_args()

# Use the parsed arguments
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_users = args.num_users
fraction = args.fraction
transmission_probability = args.transmission_probability
num_slots = args.num_slots
num_timeframes = args.num_timeframes
seeds_for_avg = args.seeds
gamma_momentum = args.gamma_momentum
num_channel_sims = args.num_channel_sims
use_memory_matrix = args.use_memory_matrix.lower() == 'true'

# Example output to ensure arguments are parsed correctly
print(f"Learning Rate: {learning_rate}")
print(f"Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print(f"Number of Users: {num_users}")
print(f"Fraction: {fraction}")
print(f"Transmission Probability: {transmission_probability}")
print(f"Number of Slots: {num_slots}")
print(f"Number of Timeframes: {num_timeframes}")
print(f"Seeds: {seeds_for_avg}")
print(f"Gamma Momentum: {gamma_momentum}")
print(f"Number of Channel Simulations: {num_channel_sims}")
print(f"Use of memory matrix: {use_memory_matrix}")

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
    print("total number of parameters:", len(tmp_list))
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

def calculate_gradient_difference(w_before, w_after):
    return [np.subtract(w_after[k], w_before[k]) for k in range(len(w_after))]

classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
num_classes = len(classes)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

size_of_user_ds = int(len(X_train)/number_of_users)
train_data_X = np.zeros((number_of_users, size_of_user_ds, 32, 32, 3))
train_data_Y = np.ones((number_of_users, size_of_user_ds, 10))
for i in range(number_of_users):
    train_data_X[i] = X_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    train_data_Y[i] = Y_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]

# Additional settings for the new requirements
num_active_users_range = range(1, 11)

# Store results
results = []
record = []
num_active_users_record = np.zeros((len(seeds_for_avg), 15))

# Initialize matrices to save gradient magnitudes
loc_grad_mag = np.zeros((len(seeds_for_avg), 15, 10))      # Local gradient magnitudes
global_grad_mag = np.zeros((len(seeds_for_avg), 15, 10))   # Global gradient magnitudes

# Dictionary to store accuracy distributions
accuracy_distributions = {seed: {timeframe: {num_active_users: [] for num_active_users in num_active_users_range} for timeframe in range(number_of_timeframes)} for seed in seeds_for_avg}

# Store mean and variance of correctly received packets
correctly_received_packets_stats = {seed: {timeframe: {num_active_users: {'mean': [], 'variance': []} for num_active_users in num_active_users_range} for timeframe in range(number_of_timeframes)} for seed in seeds_for_avg}

seed_count = 1
for seed in seeds_for_avg:
    print("************ Seed " + str(seed_count) + " ************")
    seed_count += 1
    rnd.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = tf.keras.Sequential([
        model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    for timeframe in range(num_timeframes):
        print("******** Timeframe " + str(timeframe + 1) + " ********")
        w_before_train = model.get_weights()
        model.set_weights(w_before_train)

        # Initialization of memory matrix
        memory_matrix = [[np.zeros_like(weight) for weight in w_before_train] for _ in range(num_users)]
        sparse_gradient = [[np.zeros_like(weight) for weight in w_before_train] for _ in range(num_users)]

        _, initial_accuracy = model.evaluate(X_test, Y_test)

        # List to store gradients for each user
        user_gradients = []

        # Train each user and calculate gradients
        for user_id in range(num_users):
            print("User: " + str(user_id + 1))
            model.set_weights(w_before_train)
            X_train_u = train_data_X[user_id]
            Y_train_u = train_data_Y[user_id]
            np.random.seed(5)
            shuffler = np.random.permutation(len(X_train_u))
            X_train_u = X_train_u[shuffler]
            Y_train_u = Y_train_u[shuffler]
            X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)

            history = model.fit(x=X_train_u, y=Y_train_u,
                                epochs=epochs,
                                batch_size=batch,
                                validation_data=(X_validation_u, Y_validation_u),
                                shuffle=False)
            w_after_train = model.get_weights()
            gradient_diff = calculate_gradient_difference(w_before_train, w_after_train)
            gradient_diff_memory = [gradient_diff[j] + memory_matrix[user_id][j] for j in range(len(gradient_diff))]
            if use_memory_matrix:
              sparse_gradient[user_id] = top_k_sparsificate_model_weights(gradient_diff_memory, fraction[0])
            else:
              sparse_gradient[user_id] = top_k_sparsificate_model_weights(gradient_diff, fraction[0])
            for j in range(len(w_before_train)):
                memory_matrix[user_id][j] = gamma_momentum[0] * memory_matrix[user_id][j] + gradient_diff_memory[j] - sparse_gradient[user_id][j]
            gradient_l2_norm = np.linalg.norm([np.linalg.norm(g) for g in gradient_diff])
            user_gradients.append((user_id, gradient_l2_norm, gradient_diff))

            # Save local gradient magnitude
            loc_grad_mag[seed_count - 2, timeframe, user_id] = gradient_l2_norm

        # Sort users by gradient L2 norm
        user_gradients.sort(key=lambda x: x[1], reverse=True)

        best_num_active_users = 1

        # Evaluate each number of active users to find the best one
        accuracy_sims = []
        for num_active_users in num_active_users_range:
            print("*** " + str(num_active_users) + " Active User(s) ***")
            top_users = user_gradients[:num_active_users]
            tx_prob = 1 / num_active_users

            accuracies = []
            successful_packets = []
            for _ in range(num_channel_sims):
                sum_terms = [np.zeros_like(w) for w in w_before_train]
                packets_received = 0
                for _ in range(num_slots[0]):
                    successful_users = simulate_transmissions(num_active_users, tx_prob)
                    if successful_users:
                        success_user = successful_users[0]
                        sum_terms = [np.add(sum_terms[j], sparse_gradient[success_user][j]) for j in range(len(sum_terms))]
                        packets_received += 1

                update = [np.divide(u, num_active_users) for u in sum_terms]
                new_weights = [np.add(w_before_train[i], update[i]) for i in range(len(w_before_train))]
                model.set_weights(new_weights)
                _, accuracy = model.evaluate(X_test, Y_test)
                accuracies.append(accuracy)
                successful_packets.append(packets_received)

            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            accuracy_sims.append(mean_accuracy)

            results.append({
                'seed': seed,
                'timeframe': timeframe,
                'num_active_users': num_active_users,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
            })

            # Store the accuracy distribution
            accuracy_distributions[seed][timeframe][num_active_users] = accuracies

            # Calculate and save global gradient magnitude for this number of active users
            update_l2_norm = np.linalg.norm([np.linalg.norm(g) for g in update])
            global_grad_mag[seed_count - 2, timeframe, num_active_users - 1] = update_l2_norm

            # Store mean and variance of correctly received packets
            correctly_received_packets_stats[seed][timeframe][num_active_users]['mean'] = np.mean(successful_packets)
            correctly_received_packets_stats[seed][timeframe][num_active_users]['variance'] = np.var(successful_packets)

            # Select num_active_usr for the next timeframe and use that model
            max_accuracy = np.max(accuracy_sims)
            if accuracy_sims[-1] >= max_accuracy:
              best_num_active_users = len(accuracy_sims)
              best_num_active_users_weights = new_weights

        print(f"Best number of active users for next timeframe: {best_num_active_users}")
        record.append(best_num_active_users)
        model.set_weights(best_num_active_users_weights)

    num_active_users_record[seed_count - 2,:] = record
    record = []

# Prepare data for saving
results_df = pd.DataFrame(results)

# Show the optimal number of active users throughout the timeframes
print(num_active_users_record)

# Print optimal_num_active_users, loc_grad_mag, and global_grad_mag
print("\nLocal Gradient Magnitudes:")
print(loc_grad_mag)

print("\nGlobal Gradient Magnitudes:")
print(global_grad_mag)

print("\nCorrectly Received Packets Statistics:")
print(correctly_received_packets_stats)

# Save results to a CSV file in the "FL research" folder in Google Drive
file_path = '/content/drive/My Drive/FL research/optimal_num_active_users_results_10slots.csv'
results_df.to_csv(file_path, index=False)
print(f"Results saved to: {file_path}")

# Save accuracy distributions
distributions_file_path = '/content/drive/My Drive/FL research/Distribution files for channel sims/accuracy_distributions_10slots.csv'
with open(distributions_file_path, 'w') as f:
    for seed, timeframe_data in accuracy_distributions.items():
        for timeframe, num_active_users_data in timeframe_data.items():
            for num_active_users, accuracies in num_active_users_data.items():
                f.write(f'{seed},{timeframe},{num_active_users},{",".join(map(str, accuracies))}\n')
print(f"Accuracy distributions saved to: {distributions_file_path}")

# Save correctly received packets statistics
packets_stats_file_path = '/content/drive/My Drive/FL research/correctly_received_packets_stats_10slots.csv'
with open(packets_stats_file_path, 'w') as f:
    for seed, timeframe_data in correctly_received_packets_stats.items():
        for timeframe, num_active_users_data in timeframe_data.items():
            for num_active_users, stats in num_active_users_data.items():
                mean = stats['mean']
                variance = stats['variance']
                f.write(f'{seed},{timeframe},{num_active_users},{mean},{variance}\n')
print(f"Correctly received packets statistics saved to: {packets_stats_file_path}")

# Process results to find the optimal number of active users
optimal_num_active_users = {}

for result in results:
    key = (result['seed'], result['timeframe'])
    if key not in optimal_num_active_users:
        optimal_num_active_users[key] = []
    optimal_num_active_users[key].append(result['mean_accuracy'])

for key, accuracies in optimal_num_active_users.items():
    max_accuracy = np.max(accuracies)
    optimal_num_active_users[key] = max_accuracy

print("Optimal number of active users and their accuracies:")
for key, accuracy in optimal_num_active_users.items():
    print(f"Seed: {key[0]}, Timeframe: {key[1]}, Accuracy: {accuracy:.4f}")

# Adjust the timeframes to start from 1 to 15
adjusted_optimal_num_active_users = {(seed, tf+1): acc for (seed, tf), acc in optimal_num_active_users.items()}

# Prepare data for averaging
timeframes = sorted(set(tf for _, tf in adjusted_optimal_num_active_users.keys()))
seeds = sorted(set(seed for seed, _ in adjusted_optimal_num_active_users.keys()))

# Initialize a dictionary to store the sum of accuracies and count for each timeframe
average_accuracies = {tf: [] for tf in timeframes}

# Populate the dictionary
for (seed, timeframe), accuracy in adjusted_optimal_num_active_users.items():
    average_accuracies[timeframe].append(accuracy)

# Compute the average accuracy for each timeframe
average_accuracies = {tf: np.mean(acc_list) for tf, acc_list in average_accuracies.items()}

# Plot the average accuracy vs. number of timeframes
timeframes = list(average_accuracies.keys())
avg_accuracies = list(average_accuracies.values())

plt.figure(figsize=(10, 6))
plt.plot(timeframes, avg_accuracies, marker='o', linestyle='-', color='b', label='Global Average Accuracy')
plt.xlabel('Number of Timeframes')
plt.ylabel('Average Accuracy')
plt.title('Global Accuracy Over Number Timeframes(avg = 5, without gamma_momentum)')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.xticks(timeframes)  # Ensure the x-ticks correspond to the timeframes 1-15
plt.show()
