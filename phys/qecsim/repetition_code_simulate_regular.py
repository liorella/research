## The simplest implementation of a repetition code using stim.
import numpy as np

import stim
import pymatching
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

loading = False
saving = True
##
if loading:
    loaded_data = np.load('regular_repetition_stim.npz')
    results = loaded_data.f.results
    distance_vec = loaded_data.f.distance_vec
    p_vec = loaded_data.f.p_vec
else:
    shots = 20000
    p_vec = np.logspace(-2, -0.5, num=20)
    distance_vec = range(3, 11, 2)
    results = np.zeros((len(p_vec),len(distance_vec)))
    for j, p in tqdm(enumerate(p_vec)):
        for k, distance in enumerate(distance_vec):
            rounds = distance
            circuit = stim.Circuit.generated(
                "repetition_code:memory",
                rounds=rounds,
                distance=distance,
                after_clifford_depolarization=p,
                before_round_data_depolarization=p,
                before_measure_flip_probability=p,
                after_reset_flip_probability=p
                )
            model = circuit.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(model)
            sampler = circuit.compile_detector_sampler()
            syndrome, actual_observables = sampler.sample(shots=shots, separate_observables=True)

            num_errors = 0
            for i in range(shots):
                predicted_observables = matching.decode(syndrome[i, :])
                num_errors += not np.array_equal(actual_observables[i, :], predicted_observables)
            log_error_prob=num_errors/shots
            # print(log_error_prob)
            results[j,k]=log_error_prob
    if saving:
        np.savez('regular_repetition_stim.npz', results=results, distance_vec=distance_vec, p_vec=p_vec)


## plotting
fig, ax = plt.subplots()
for i, distance in enumerate(distance_vec):
    ax.plot(p_vec, results[:,i], 'o', label=f'distance={distance}')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('physical error')
plt.ylabel('logical error')
plt.show()
plt.xlim([1E-2,p_vec[-1]+0.1])
plt.ylim([0.5E-3, 0.6])
plt.grid(linestyle='--', linewidth=0.2)

plt.savefig('regular_repetition_stim.svg')
