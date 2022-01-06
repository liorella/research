import quantumsim  # https://gitlab.com/quantumsim/quantumsim, branch: stable/v0.2
import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm

from qecsim.rep_code_generator import RepCodeGenerator, CircuitParams

log = logging.getLogger('qec')
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# run simulation
########

plot = False
distance = 2
encoded_data = True
num_rounds = 20
num_max_iterations = 500  # to maintain reasonable time we do a constant number of rounds * iterations
distance_vec = np.arange(2, 5, 1)
rounds_vec = np.arange(1, 20, 4)

success_rate_matrix = []
success_sigma_matrix = []
for distance in distance_vec:
    cparams = CircuitParams(t1=5e3,
                            t2=5e3,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=40,
                            meas_duration=200,
                            reset_duration=20,
                            reset_latency=0)

    repc = RepCodeGenerator(distance=distance,
                            circuit_params=cparams
                            )

    stabilizer = repc.generate_stabilizer_round(plot=plot)

    # start cycle
    success_rate_vector = []
    success_sigma_vector = []
    for num_rounds in tqdm(rounds_vec):
        events_fraction = np.zeros(num_rounds + 1)
        success_vector = []
        for n in range(num_max_iterations // num_rounds):
            state = quantumsim.sparsedm.SparseDM(repc.register_names)
            syndromes = []
            if encoded_data:
                repc.generate_logical_X(plot=plot).apply_to(state)

            # repc.generate_bitflip_error('0').apply_to(state)  # for testing purposes

            for i in range(num_rounds - 1):
                stabilizer.apply_to(state)
                syndromes.append([state.classical[cb] for cb in repc.cbit_names[1::2]])
                # apply active reset
                to_reset = []
                for q, cb in zip(repc.qubit_names[1::2], repc.cbit_names[1::2]):
                    if state.classical[cb] == 1:
                        to_reset.append(q)
                repc.generate_active_reset(to_reset).apply_to(state)

            repc.generate_stabilizer_round(final_round=True, plot=plot).apply_to(state)
            syndromes.append([state.classical[cb] for cb in repc.cbit_names[1::2]])
            data_meas = np.array([state.classical[cb] for cb in repc.cbit_names[::2]])

            # postprocessing
            data_meas_parity = repc.matching_matrix @ data_meas % 2
            # we prepend zeros to account for first round and append perfect measurement step parity
            syndromes = np.vstack([np.zeros(distance), syndromes, data_meas_parity])
            detection_events = np.logical_xor(syndromes[1:], syndromes[:-1])
            log.debug("detection events")
            log.debug("\n" + repr(detection_events.astype(int).T))
            pauli_frame = Matching(repc.matching_matrix, repetitions=detection_events.shape[0]).decode(detection_events.T)
            log.debug("Pauli frame")
            log.debug(pauli_frame)
            log.debug("data qubits meas result")
            log.debug(data_meas)
            recovered = np.logical_xor(data_meas, pauli_frame)
            log.debug("recovered state")
            log.debug(recovered.astype(int))
            success = not np.any(np.logical_xor(recovered, [encoded_data] * len(recovered)))
            log.debug(f"success = {success}")
            success_vector.append(success)
            events_fraction = n / (n + 1) * events_fraction + 1 / (n + 1) * detection_events.mean(1)
        success_rate = np.array(success_vector).mean()
        success_sigma = np.sqrt(success_rate * (1 - success_rate) / len(success_vector))  # binomial distribution
        success_rate_vector.append(success_rate)
        success_sigma_vector.append(success_sigma)
    success_rate_matrix.append(success_rate_vector)
    success_sigma_matrix.append(success_sigma_vector)

success_rate_matrix = np.array(success_rate_matrix)
success_sigma_matrix = np.array(success_sigma_matrix)

print("events fraction")
print(events_fraction)
for i in range(success_rate_matrix.shape[0]):
    plt.errorbar(rounds_vec, success_rate_matrix[i], yerr=success_sigma_matrix[i], label=f"distance {distance_vec[i]}")
plt.xlabel('number of rounds')
plt.ylabel('success rate')
plt.legend()
plt.show()
