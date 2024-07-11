import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb

# Function to calculate the probability of k unique users being decoded over m slots
def probability_k_unique_users(n, m, p, k):
    # Slot success probability
    P_s = n * p * (1 - p)**(n - 1)
    
    if k == 0:
        # Probability that no users are decoded is the probability that all m slots fail
        return (1 - P_s) ** m
    else:
        total_prob = 0
        for s in range(1, m + 1):
            # Binomial coefficient (m choose s)
            comb_ms = comb(m, s)
            # Probability of s successful transmissions
            success_prob = P_s**s
            # Probability of m-s failed transmissions
            failure_prob = (1 - P_s)**(m - s)
            # Binomial coefficient (n choose k)
            choose_k_from_n = comb(n, k)
            # Ways to assign s successes to k users, allowing repetitions
            repeated_selections = k**s

            # Calculate the term for this s
            term = (comb_ms * success_prob * failure_prob * choose_k_from_n * repeated_selections) / (n**s)
            total_prob += term

        return total_prob

# Parameters
n = 10  # Number of users
m = 10  # Number of slots
p = 0.5  # Probability of transmission per slot
k_values = range(0, n + 1)  # Different k values to evaluate
num_timeframes = 15  # Number of timeframes
num_simulations_2 = 1000  # Number of simulations for method 2

# Analytical Method
# Calculate probabilities for different k values
probabilities_analytical = [probability_k_unique_users(n, m, p, k) for k in k_values]

# Normalize probabilities to ensure they sum up to 1
prob_sum = sum(probabilities_analytical)
probabilities_analytical = [prob / prob_sum for prob in probabilities_analytical]

# Simulation Method
def simulate_transmissions(number_of_users, transmission_probability):
    decisions = np.random.rand(number_of_users) < transmission_probability
    if np.sum(decisions) == 1:
        successful_users = [i for i, decision in enumerate(decisions) if decision]
    else:
        successful_users = []
    return successful_users

# Function to simulate multiple timeframes using the second method
def simulate_multiple_timeframes_method_2(num_timeframes):
    results = []

    for timeframe in range(num_timeframes):
        num_active_users = 3
        tx_prob = 1 / num_active_users
        successful_transmissions = []

        for _ in range(m):
            successful_users = simulate_transmissions(num_active_users, tx_prob)
            if successful_users:
                successful_transmissions.append(successful_users[0])
            else:
                successful_transmissions.append(None)

        results.append(len([user for user in successful_transmissions if user is not None]))

    return results

# Run the simulations for method 2
all_results_method_2 = []

for _ in range(num_simulations_2):
    all_results_method_2.extend(simulate_multiple_timeframes_method_2(num_timeframes))

# Convert results to DataFrame for analysis
df_method_2 = pd.DataFrame(all_results_method_2, columns=['Successful Transmissions'])

# Calculate probabilities for the simulation method
probabilities_simulation = df_method_2['Successful Transmissions'].value_counts(normalize=True).sort_index()

# Plot the results side by side
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.bar(k_values, probabilities_analytical, alpha=0.7, color='blue', label='Analytical Method')
plt.xlabel('Number of Unique Users Decoded (k)')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.title('Analytical Method')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(probabilities_simulation.index, probabilities_simulation.values, alpha=0.7, color='green', label='Simulation Method')
plt.xlabel('Number of Unique Users Decoded (k)')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.title('Simulation Method')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print the calculated probabilities for both methods
print("Analytical Method:")
for k, prob in zip(k_values, probabilities_analytical):
    print(f'Probability of decoding {k} unique users: {prob:.4f}')

print("\nSimulation Method:")
for k in range(0, m + 1):
    prob = probabilities_simulation.get(k, 0)
    print(f'Probability of decoding {k} unique users: {prob:.4f}')

# Summary statistics for the simulation method
summary_method_2 = df_method_2['Successful Transmissions'].describe()
print("\nMethod 2: Distribution of Successful Transmissions")
print(summary_method_2)
