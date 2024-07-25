import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
from scipy.stats import binom

# Number of users
n = 10
# Number of slots per frame
m = 10
# Transmission probability in each slot
p = 1 / n

# Function to compute Stirling numbers of the second kind
def stirling_second_kind(n, k):
    S = 0
    for i in range(k + 1):
        S += (-1) ** (k - i) * comb(k, i) * i ** n
    return S / factorial(k)

# Simulate the SA channel and collect the number of different users decoded
num_users = np.zeros(int(1e5))

for ss in range(int(1e5)):
    tx_mat = np.random.rand(m, n)
    tx_mat[tx_mat <= p] = 1
    tx_mat[tx_mat < 1] = 0
    outcomes = np.sum(tx_mat, axis=1)
    succ_id = np.where(outcomes == 1)[0]
    dec_nodes = []

    for ii in succ_id:
        dec_nodes.extend(np.where(tx_mat[ii, :] == 1)[0])

    num_users[ss] = len(np.unique(dec_nodes))

# PMF of the number of different users collected in a frame, via sims
plt.figure()
aa = plt.hist(num_users, bins=np.arange(0, m+2)-0.5, density=True, alpha=0.6, color='g', edgecolor='black')

# Analytical
ps = n * (1 / n) * (1 - 1 / n) ** (n - 1)
pmf_succ = binom.pmf(np.arange(1, m+1), m, ps)

cond_pmf = np.zeros((m, m))
for s in range(1, m+1):
    for k in range(1, s+1):
        cond_pmf[k-1, s-1] = comb(n, k) * stirling_second_kind(s, k) * factorial(k) / (n ** s)

pmf_teo = np.dot(pmf_succ, cond_pmf.T)

# Include the probability of decoding 0 users
pmf_teo = np.insert(pmf_teo, 0, (1 - ps) ** m)

# Plot the analytical PMF
plt.stem(np.arange(0, m+1), pmf_teo, linefmt='r-', markerfmt='ro', basefmt='r-')

plt.title('PMF of the Number of Different Users Decoded in a Frame')
plt.xlabel('Number of Users')
plt.ylabel('Probability')
plt.legend(['Analytical', 'Simulation'])
plt.grid(True)
plt.show()
