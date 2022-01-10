# timing guide for M1 mac:
# distance 2 - 3.4 msec per iteration * round
# distance 3 - 7 msec per iteration * round
# distance 4 - 18 msec per iteration * round
from itertools import cycle

import quantumsim  # https://gitlab.com/quantumsim/quantumsim, branch: stable/v0.2
import numpy as np
import qutip
from pymatching import Matching
import matplotlib.pyplot as plt
import logging

from tqdm import tqdm

from qecsim.lib import quantumsim_dm_to_qutip_dm
from qecsim.rep_code_generator import RepCodeGenerator, CircuitParams

# run simulation
########

plot = False
log = logging.getLogger('qec')
log.setLevel(logging.INFO)
# distance = 4
encoded_state = '+'

if encoded_state == '1':
    expected_prob = 1
elif encoded_state == '0':
    expected_prob = 0
else:
    expected_prob = 0.5

# num_rounds = 20
num_max_iterations = int(1e4)
distance_vec = np.arange(2, 4, 1)
rounds_vec = np.arange(1, 10, 2)

logical_1_prob_matrix = []
success_sigma_matrix = []
log.info("starting simulation")
for distance in distance_vec:
    log.info(f"distance = {distance}")
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

    # start cycle
    stabilizer = repc.generate_stabilizer_round(plot=plot)
    logical_1_prob_vector = []
    logical_1_sigma_vector = []
    for num_rounds in rounds_vec:
        log.info(f'rounds = {num_rounds}')
        events_fraction = np.zeros(num_rounds + 1)
        log_state_outcome_vector = []
        for n in tqdm(range(num_max_iterations)):
            state = quantumsim.sparsedm.SparseDM(repc.register_names)
            syndromes = []

            repc.generate_state_encoder(encoded_state, plot=plot).apply_to(state)
            if plot and distance == 2:  # currently hardcoded to this distance
                state.renormalize()
                qdm = quantumsim_dm_to_qutip_dm(state)
                data_qdm = qdm.ptrace(range(2, 5))
                qutip.matrix_histogram_complex(data_qdm)
                plt.title('data qubits DM')
                plt.show()
            # for i in range(0, 4, 2):
            #     repc.generate_bitflip_error(str(i), plot=plot).apply_to(state)  # for testing purposes

            for i in range(num_rounds - 1):
                stabilizer.apply_to(state)
                syndromes.append([state.classical[cb] for cb in repc.cbit_names[1::2]])
                # apply active reset
                to_reset = []
                for q, cb in zip(repc.qubit_names[1::2], repc.cbit_names[1::2]):
                    if state.classical[cb] == 1:
                        to_reset.append(q)
                repc.generate_active_reset(to_reset).apply_to(state)

            if plot and distance == 2:  # currently hardcoded to this distance
                state.renormalize()
                qdm = quantumsim_dm_to_qutip_dm(state)
                data_qdm = qdm.ptrace(range(2, 5))
                qutip.matrix_histogram_complex(data_qdm)
                plt.title(f'data qubits DM after {num_rounds} rounds')
                plt.show()
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
            recovered = np.logical_xor(data_meas, pauli_frame).astype(int)
            log.debug("recovered state")
            log.debug(recovered)
            assert np.all(recovered) == recovered[0], f"decoder failed - recovered value is {recovered}"
            log_state_outcome = recovered[0]
            log.debug(f"logical state outcome = {log_state_outcome}")
            log_state_outcome_vector.append(log_state_outcome)
            events_fraction = n / (n + 1) * events_fraction + 1 / (n + 1) * detection_events.mean(1)
        logical_1_prob = np.array(log_state_outcome_vector).mean()
        logical_1_sigma = np.sqrt(logical_1_prob * (1 - logical_1_prob) / len(log_state_outcome_vector))  # binomial distribution
        logical_1_prob_vector.append(logical_1_prob)
        logical_1_sigma_vector.append(logical_1_sigma)
    logical_1_prob_matrix.append(logical_1_prob_vector)
    success_sigma_matrix.append(logical_1_sigma_vector)

logical_1_prob_matrix = np.array(logical_1_prob_matrix)
success_sigma_matrix = np.array(success_sigma_matrix)
trace_distance_matrix = np.abs(expected_prob - logical_1_prob_matrix)
log.info("simulation done")

print("events fraction")
print(events_fraction)
for i in range(logical_1_prob_matrix.shape[0]):
    plt.errorbar(rounds_vec, trace_distance_matrix[i], yerr=success_sigma_matrix[i], label=f"distance {distance_vec[i]}")
plt.title(f'encoded state =|{encoded_state}>')
plt.xlabel('number of rounds')
plt.ylabel('trace distance')
plt.legend()
plt.grid('all')
plt.show()
