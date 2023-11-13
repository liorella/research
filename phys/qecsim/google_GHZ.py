
import numpy as np
from qec_generator import CircuitParams
from stim_lib.scheduled_circuit import generate_scheduled
from matplotlib import pyplot as plt
import stimcirq

distance_vec=range(3,7,2)
shots = 50000


## via swap
cparams = CircuitParams(t1=6e3,
                t2=6e3,
                single_qubit_gate_duration=20,
                two_qubit_gate_duration=40,
                single_qubit_depolarization_rate=1.2e-3,
                two_qubit_depolarization_rate=7.5e-3,
                meas_duration=350,
                reset_duration=20,
                reset_latency=300,
                measurement_error=2E-2,
                meas_induced_dephasing_enhancement=1)



GHZ_probability = np.zeros((len(distance_vec),1))
for k, distance in enumerate(distance_vec):
    circ, cont, _ = generate_scheduled(
        code_task='GHZ_circuit',
        distance=distance,
        rounds=0,
        params=cparams,
    )
    success = 0
    sampler = circ.compile_detector_sampler()
    results = sampler.sample(shots=shots)
    sum_results=results.sum(axis=1) % distance
    GHZ_probability[k]=1-(sum(sum_results!=0) / shots)
##
plt.plot(distance_vec, GHZ_probability, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('number of qubits')
plt.text(-0.5, 0.5, r'probability that $|0\rangle^{\otimes n}$ or $|1\rangle^{\otimes n}$ was measured', rotation=90, va='center', ha='center')


# Display the plot
plt.show()
