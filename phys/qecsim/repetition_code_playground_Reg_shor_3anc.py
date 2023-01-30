## The simplest implementation of a repetition code using stim.
import numpy as np

import stim
import pymatching
import matplotlib.pyplot as plt
import networkx as nx

from stim_lib.run_feedback import measure_instructions

from qec_generator import CircuitParams
from stim_lib.error_context import StimErrorContext
from stim_lib.scheduled_circuit import generate_scheduled, get_pauli_probs, instruction_duration
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results
from qec_generator import CircuitParams
from simulate_qec_rounds_stim import experiment_run
from stim_lib.scheduled_circuit import generate_scheduled
from simulate_qec_rounds_stim import gen_feedback_circuit
from simulate_qec_rounds_stim import gen_syndrome_circuit
from simulate_qec_rounds_stim import gen_anc_parity_circuit

from numpy.linalg import inv

from scipy.optimize import curve_fit
##

rounds = 6
distance = 7
p = 0.01

cparams = CircuitParams(t1=0, #if t1=0 than use the single probability
                t2=0,
                single_qubit_gate_duration=20,
                two_qubit_gate_duration=20,
                single_qubit_depolarization_rate=p,
                two_qubit_depolarization_rate=p,
                meas_duration=550,
                reset_duration=0,
                reset_latency=40,
                meas_induced_dephasing_enhancement=3)

circuit, context , _ = generate_scheduled(
                code_task='shor_style_3anc_syndrome_extraction',  # looks ok
                distance=distance,
                rounds=rounds,
                t1_t2_depolarization=False,
                meas_induced_dephasing_enhancement=False,
                params=cparams
                ) # does not include before_round_data_depolarization=p and before_measure_flip_probability=p
# The context include the matching matrix

## defining manually without touching the context

num_rounds = rounds
anc1 = range(4 * (distance - 1) + 1)[1::4]
anc2 = range(4 * (distance - 1) + 1)[3::4]
anc3 = range(4 * (distance - 1) + 1)[2::4]
active_ancillas = np.vstack((anc1, anc2)).T.flatten()
ancillas_to_correct_in_parity = np.vstack((anc3, anc2)).T.flatten()

data_qubits = (range(4 * (distance - 1) + 1))[::4]

matching_matrix=np.zeros((distance-1, distance), dtype=np.uint8)
for r in range(distance-1):
    matching_matrix[r, r] = 1
    matching_matrix[r, r+1] = 1

synd2data = matching_matrix.T @ inv(matching_matrix@matching_matrix.T % 2)

segment2 = gen_syndrome_circuit(data_qubits, anc1, anc2, context)

segment3 = gen_anc_parity_circuit(anc1, anc2, anc3, context)

t = np.floor((distance-1)/2)
## running the exp.
success = 0
shots = 1000
rounds_count = rounds*np.ones(shots)

for shot in range(shots):
    reset_indicator = np.zeros(len(active_ancillas), dtype=np.uint8)
    sim = stim.TableauSimulator() #neccesary to add components into the circuit
    success_counter = 0
    fail_counter = 0
    rep_circ_iter = to_measure_segments(circuit)  # breaking down the circuit into segments, divided by measurement events
    results_prev = np.zeros(distance-1)
    for i, segment in enumerate(rep_circ_iter):
        # print(i)
        # break
        if (i < num_rounds * 2) and (i % 2 ==0):
            sim.do(gen_feedback_circuit(reset_indicator, active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
            parity_ancilla_mes = do_and_get_measure_results(sim, segment, anc3)  # simulate the segment
            anc_parity_reset_indicator = np.vstack((parity_ancilla_mes, parity_ancilla_mes)).T.flatten()
            sim.do(gen_feedback_circuit(anc_parity_reset_indicator, ancillas_to_correct_in_parity, context)) # gives an X gate for the ancilla the is to reset and append errors
        elif i < num_rounds * 2:
            ancilla_mes = do_and_get_measure_results(sim, segment, active_ancillas)  # simulate the segment
            results = (np.diff(ancilla_mes)%2)[::2]
            if (results == results_prev).all():
                success_counter += 1
                if success_counter == t+1:
                    success_counter = 0
                    fail_counter = 0
                    data_to_flip = (synd2data @ results) % 2
                    if np.sum(data_to_flip) > t:
                        data_to_flip = (data_to_flip+1) % 2
                    sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
            else:
                fail_counter+=1
                success_counter=0
                if fail_counter == (t+1)**2:
                    fail_counter = 0
                    data_to_flip = (synd2data @ results).astype(np.uint8)
                    if np.sum(data_to_flip)>t:
                        data_to_flip = (data_to_flip+1) % 2
                    sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
            reset_indicator = ancilla_mes
            results_prev = results
#            print(i)
 #           print(results)
        elif i == num_rounds * 2:
         #  break
            while 0 < success_counter < t+1 or 0 < fail_counter < (t+1)**2:
                sim.do(gen_feedback_circuit(reset_indicator, active_ancillas,
                                            context))  # gives an X gate for the ancilla the is to reset and append errors
                parity_ancilla_mes = do_and_get_measure_results(sim, segment3, anc3)  # simulate the segment
                anc_parity_reset_indicator = np.vstack((parity_ancilla_mes, parity_ancilla_mes)).T.flatten()
                sim.do(gen_feedback_circuit(anc_parity_reset_indicator, ancillas_to_correct_in_parity,
                                            context))  # gives an X gate for the ancilla the is to reset and append errors
                rounds_count[shot]+=1
                ancilla_mes = do_and_get_measure_results(sim, segment2, active_ancillas)
                results = (np.diff(ancilla_mes) % 2)[::2]
                if (results == results_prev).all():
                    success_counter += 1
                    if success_counter == t + 1:
                        fail_counter = 0
                        data_to_flip = (synd2data @ results) % 2
                        if np.sum(data_to_flip) > t:
                            data_to_flip = (data_to_flip + 1) % 2
                        sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                else:
                    fail_counter += 1
                    success_counter = 0
                    if fail_counter == (t + 1) ** 2:
                        data_to_flip = (synd2data @ results).astype(np.uint8)
                        if np.sum(data_to_flip) > t:
                            data_to_flip = (data_to_flip + 1) % 2
                        sim.do(gen_feedback_circuit(data_to_flip, np.array(data_qubits), context))
                results_prev = results
                reset_indicator = ancilla_mes
            results_data = do_and_get_measure_results(sim, segment, data_qubits)
            logical_state = np.around(np.sum(results_data)/context.distance, 0)
            if logical_state == 0:
                success += 1

log_error_prob=1-success/shots
print(log_error_prob)

print(np.mean(rounds_count))
