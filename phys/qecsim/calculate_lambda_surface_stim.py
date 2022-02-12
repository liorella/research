import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from qecsim.qec_generator import CircuitParams
from qecsim.simulate_qec_rounds_stim import experiment_run_apt
from qecsim.stim_lib.scheduled_circuit import generate_scheduled

distance_vec = [3, 5, 7]
rounds_vec = np.arange(2, 20, 4)
success_rate = []

cparams = CircuitParams(t1=20e3,
                        t2=20e3,
                        single_qubit_gate_duration=20,
                        two_qubit_gate_duration=100,
                        single_qubit_depolarization_rate=0,
                        two_qubit_depolarization_rate=0,
                        meas_duration=600,
                        reset_duration=0,
                        reset_latency=40)

task = 'surface_code:rotated_memory_z'  # looks ok
# task = 'surface_code:unrotated_memory_z'  # looks ok

for distance in distance_vec:
    success_rate.append([])
    print(f'starting distance = {distance}')
    for rounds in tqdm(rounds_vec):
        circ, cont = generate_scheduled(
            code_task='surface_code:rotated_memory_z',  # looks ok
            distance=distance,
            rounds=rounds,
            params=cparams
        )
        success_rate[-1].append(experiment_run_apt(circ, cont, shots=1000))

success_rate = np.array(success_rate)
for i, distance in enumerate(distance_vec):
    plt.plot(rounds_vec, success_rate[i], label=f'distance={distance}')
plt.legend()
plt.show()
