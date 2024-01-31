# %%
# this code calculates by how much we can lower the error of single shot readout with a 50% 50% a priori probability of 0 and 1
# given a certain acceptance threshold
import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

Nshots = 100000
mu = 2


def Q(x): return 0.5*erfc(x/np.sqrt(2))


def sample(mu, Nshots, acceptance_threshold):
    noise = np.random.randn(Nshots)
    result_bit = np.random.rand(Nshots) < 0.5
    measured_result = mu*(2*result_bit - 1) + noise
    accepted = np.abs(measured_result) > acceptance_threshold
    acceptance_rate = np.sum(accepted) / Nshots
    error_rate = np.sum(accepted & (
        result_bit ^ (measured_result > 0))) / Nshots
    return measured_result, error_rate, acceptance_rate


measured_result, error_rate, acceptance_rate = sample(mu, Nshots, 0)
plt.hist(measured_result, bins=100)
plt.title(f"Error rate: {error_rate}, Q={Q(mu):.4f}")

# %%

mu_vec = np.linspace(0, 4, 100)
plt.semilogy(mu_vec, 0.5*erfc(mu_vec/np.sqrt(2)), label="Theory")
plt.semilogy(mu_vec, [sample(mu, Nshots, 0)[1]
             for mu in mu_vec], label="Simulation")
plt.legend()
# %% same, now with acceptance-rejection sampling

mu_vec = np.linspace(1.5, 2, 3)
r_vec = np.linspace(0.0, 4, 50)
error_rate = np.zeros((len(mu_vec), len(r_vec)))
acceptance_rate = np.zeros((len(mu_vec), len(r_vec)))
for i, mu in enumerate(mu_vec):
    for j, acceptance_threshold in enumerate(r_vec):
        _, error_rate[i, j], acceptance_rate[i, j] = sample(
            mu, Nshots, acceptance_threshold)

for i, mu in enumerate(mu_vec):
    plt.plot(r_vec,
             error_rate[i], f"C{i}",  label=f"mu={mu}")
    plt.plot(r_vec, Q(mu+r_vec), f"C{i}--", label=f"mu={mu}")
    plt.xlabel("Acceptance threshold")
    plt.ylabel("Error rate")

plt.xlim([0.5, 1])
plt.legend()
plt.show()

for i, mu in enumerate(mu_vec):
    plt.plot(r_vec,
             acceptance_rate[i], f"C{i}", label=f"mu={mu_vec[i]}")
    plt.plot(r_vec, Q(r_vec - mu) + Q(mu+r_vec),
             f"C{i}--", label=f"mu={mu} theory")
    plt.xlabel("Acceptance threshold")
    plt.ylabel("acceptance rate")

plt.legend()
plt.show()

for i, mu in enumerate(mu_vec):
    plt.semilogy(acceptance_rate[i], error_rate[i], f"C{i}", label=f"mu={mu_vec[i]}")
    plt.semilogy(Q(r_vec - mu) + Q(mu+r_vec), Q(mu+r_vec), f"C{i}--"),
    plt.xlabel("Acceptance rate")
    plt.ylabel("Error rate")
plt.xlim([0.5, 1])
# fit y axis limit to data
plt.ylim([1e-5, 0])
plt.legend()
plt.grid('both', 'both')
plt.title("Error rate vs acceptance rate")

# %%
