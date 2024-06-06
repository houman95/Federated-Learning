import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random as rnd
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# DNN settings
learning_rate = 0.01
epochs = 3  # Set epochs as a constant

# Change seed calculate the average
seeds_for_avg = [42, 57, 85, 12, 29, 33, 7, 91]

batch = 128
number_of_users = 10
fraction = [0.1, 0.15, 0.2, 0.4]

# Slotted ALOHA settings
transmission_probability = 1 / number_of_users
number_of_slots = [5, 10, 20]
number_of_timeframes = 15

compression_type = "no compression"

# Transformations for the CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=2)

classes = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}

# Define the model
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_validation_split(X_train, Y_train, val_fraction=0.25):
    train_length = len(X_train)
    validation_length = int(train_length * val_fraction)
    X_validation = X_train[:validation_length]
    Y_validation = Y_validation[:validation_length]
    X_train = X_train[validation_length:]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation

def top_k_sparsificate_model_weights(weights, fraction):
    tmp_list = []
    for el in weights:
        lay_list = el.reshape((-1)).tolist()
        tmp_list = tmp_list + [abs(el) for el in lay_list]
    tmp_list.sort(reverse=True)
    print("total number of parameters:", len(tmp_list))
    k_th_element = tmp_list[int(fraction*len(tmp_list))-1]
    new_weights = []
    for el in weights:
        mask = torch.ge(torch.abs(el), k_th_element)
        new_w = el * mask.float()
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
    return [w_after[k] - w_before[k] for k in range(len(w_after))]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(classes)
size_of_user_ds = int(len(trainset) / number_of_users)
train_data_X = torch.zeros((number_of_users, size_of_user_ds, 3, 32, 32))
train_data_Y = torch.ones((number_of_users, size_of_user_ds), dtype=torch.long)

train_data_X = train_data_X.to(device)
train_data_Y = train_data_Y.to(device)

train_data_X = train_data_X.to(device)
train_data_Y = train_data_Y.to(device)

for i in range(number_of_users):
    indices = list(range(size_of_user_ds * i, size_of_user_ds * (i + 1)))
    subset = torch.utils.data.Subset(trainset, indices)
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=size_of_user_ds, shuffle=False)
    for j, (data, target) in enumerate(subset_loader):
        train_data_X[i] = data
        train_data_Y[i] = target

# Additional settings for the new requirements
num_active_users_range = range(1, 11)
num_channel_sims = 100

# This is momentum for memory matrix
gamma_momentum = [1, 0.9, 0.8, 0.7, 0.5, 0.1]

# Store results
results = []
record = []
num_active_users_record = np.zeros((len(seeds_for_avg), 15))

# Initialize matrices to save gradient magnitudes
loc_grad_mag = np.zeros((len(seeds_for_avg), 15, 10))      # Local gradient magnitudes
global_grad_mag = np.zeros((len(seeds_for_avg), 15, 10))   # Global gradient magnitudes

# Dictionary to store accuracy distributions
accuracy_distributions = {seed: {timeframe: {num_active_users: [] for num_active_users in num_active_users_range} for timeframe in range(number_of_timeframes)} for seed in seeds_for_avg}

seed_count = 1
for seed in seeds_for_avg:
    print("************ Seed " + str(seed_count) + " ************")
    seed_count += 1
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = VGG16(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for timeframe in range(number_of_timeframes):
        print("******** Timeframe " + str(timeframe + 1) + " ********")
        w_before_train = [param.data.clone() for param in model.parameters()]
        model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), w_before_train)})

        # Initialization of memory matrix
        memory_matrix = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(number_of_users)]
        sparse_gradient = [[torch.zeros_like(param).to(device) for param in w_before_train] for _ in range(number_of_users)]

        initial_accuracy = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        initial_accuracy = 100 * correct / total

        # List to store gradients for each user
        user_gradients = []

        # Train each user and calculate gradients
        for user_id in range(number_of_users):
            print("User: " + str(user_id + 1))
            model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), w_before_train)})

            X_train_u = train_data_X[user_id]
            Y_train_u = train_data_Y[user_id]
            np.random.seed(5)
            shuffler = np.random.permutation(len(X_train_u))
            X_train_u = X_train_u[shuffler]
            Y_train_u = Y_train_u[shuffler]
            X_train_u, Y_train_u, X_validation_u, Y_validation_u = train_validation_split(X_train_u, Y_train_u)

            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_u)
                loss = criterion(outputs, Y_train_u)
                loss.backward()
                optimizer.step()

            w_after_train = [param.data.clone() for param in model.parameters()]
            gradient_diff = calculate_gradient_difference(w_before_train, w_after_train)
            gradient_diff_memory = [gradient_diff[j] + memory_matrix[user_id][j] for j in range(len(gradient_diff))]
            sparse_gradient[user_id] = top_k_sparsificate_model_weights(gradient_diff_memory, fraction[2])
            for j in range(len(w_before_train)):
                memory_matrix[user_id][j] = gamma_momentum[0] * memory_matrix[user_id][j] + gradient_diff_memory[j] - sparse_gradient[user_id][j]
            gradient_l2_norm = torch.norm(torch.stack([torch.norm(g) for g in gradient_diff])).item()
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
            for _ in range(num_channel_sims):
                sum_terms = [torch.zeros_like(param).to(device) for param in w_before_train]
                for _ in range(number_of_slots[1]):
                    successful_users = simulate_transmissions(num_active_users, tx_prob)
                    if successful_users:
                        success_user = successful_users[0]
                        sum_terms = [sum_terms[j] + sparse_gradient[success_user][j] for j in range(len(sum_terms))]

                update = [u / num_active_users for u in sum_terms]
                new_weights = [w_before_train[i] + update[i] for i in range(len(w_before_train))]
                model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), new_weights)})

                correct = 0
                total = 0
                model.eval()
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                accuracies.append(accuracy)

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
            update_l2_norm = torch.norm(torch.stack([torch.norm(g) for g in update])).item()
            global_grad_mag[seed_count - 2, timeframe, num_active_users - 1] = update_l2_norm

            # Select num_active_usr for the next timeframe and use that model
            max_accuracy = np.max(accuracy_sims)
            if accuracy_sims[-1] >= max_accuracy:
              best_num_active_users = len(accuracy_sims)
              best_num_active_users_weights = new_weights

        print(f"Best number of active users for next timeframe: {best_num_active_users}")
        record.append(best_num_active_users)
        model.load_state_dict({k: v for k, v in zip(model.state_dict().keys(), best_num_active_users_weights)})

    num_active_users_record[seed_count - 2, :] = record
    record = []

# Prepare data for saving
results_df = pd.DataFrame(results)

# Save results to a CSV file
file_path = 'optimal_num_active_users_results_10slots.csv'
results_df.to_csv(file_path, index=False)
print(f"Results saved to: {file_path}")

# Save accuracy distributions
distributions_file_path = 'accuracy_distributions_10slots.csv'
with open(distributions_file_path, 'w') as f:
    for seed, timeframe_data in accuracy_distributions.items():
        for timeframe, num_active_users_data in timeframe_data.items():
            for num_active_users, accuracies in num_active_users_data.items():
                f.write(f'{seed},{timeframe},{num_active_users},{",".join(map(str, accuracies))}\n')
print(f"Accuracy distributions saved to: {distributions_file_path}")

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
plt.legend(loc='lower right')
plt.grid(True)
plt.xticks(timeframes)  # Ensure the x-ticks correspond to the timeframes 1-15
plt.show()

# Print optimal_num_active_users, loc_grad_mag, and global_grad_mag
print("\nOptimal number of active users and their accuracies:")
for key, accuracy in optimal_num_active_users.items():
    print(f"Seed: {key[0]}, Timeframe: {key[1]}, Accuracy: {accuracy:.4f}")

print("\nLocal Gradient Magnitudes:")
print(loc_grad_mag)

print("\nGlobal Gradient Magnitudes:")
print(global_grad_mag)
