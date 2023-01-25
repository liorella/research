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

from scipy.optimize import curve_fit
##
rounds=4
distance=7
p=0.1

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
                code_task='shor_style_syndrome_extraction',  # looks ok
                distance=distance,
                rounds=rounds,
                t1_t2_depolarization=False,
                meas_induced_dephasing_enhancement=False,
                params=cparams
                ) # does not include before_round_data_depolarization=p and before_measure_flip_probability=p
# The context include the matching matrix

## breaking down the 'experiment_run' (only active reset, and nodify to majority voting)

#defining manually without touching the context

num_rounds = rounds
active_ancillas = np.vstack((range(3 * (distance - 1) + 1)[1::3], range(3 * (distance - 1) + 1)[2::3])).T.flatten()
data_qubits = (range(3 * (distance - 1) + 1))[::3]
matching_matrix=np.zeros((distance-1, distance), dtype=np.uint8)
for r in range(distance-1):
    matching_matrix[r, r] = 1
    matching_matrix[r, r+1] = 1
m = pymatching.Matching(matching_matrix, repetitions=rounds + 1)

# running the exp.
success = 0
shots = 1000

logical_counter = 0 #checks if after the decoding there is a difference between the bits

for shot in range(shots):
    rep_circ_iter = to_measure_segments(circuit) #breaking down the circuit into segments, divided by measurement events
    reset_indicator = np.zeros(len(active_ancillas), dtype=np.uint8)
    results_record = np.zeros((num_rounds + 1, distance-1), dtype=np.uint8) # a matrix that will eventually reach the decoder
    sim = stim.TableauSimulator() #neccesary to add components into the circuit

    for i, segment in enumerate(rep_circ_iter):
        if i < num_rounds:
            sim.do(gen_feedback_circuit(reset_indicator, active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
            ancilla_mes = do_and_get_measure_results(sim, segment, active_ancillas)  # simulate the segment
            results = (np.diff(ancilla_mes)%2)[::2]
            results_record[i, :] = results
            reset_indicator = ancilla_mes
#            print(i)
 #           print(results)
        elif i == num_rounds:
            results_data = do_and_get_measure_results(sim, segment, data_qubits)
            data_parity = matching_matrix @ results_data % 2
            results_record[i, :] = data_parity
            to_decode = np.diff(np.hstack((np.zeros(distance-1, dtype=np.uint8)[:, np.newaxis], results_record.T))) % 2
            frame = m.decode(to_decode)
            corrected_state = (results_data + frame) % 2
            logical_state = corrected_state[-1]
            # logical_state = np.around(np.sum(corrected_state)/context.distance, 0)
            # if np.sum(corrected_state) == 0 or np.sum(corrected_state) == distance: #only for checking if the previous line is neccesary
            #     logical_counter += 1
            if logical_state == 0:
                success += 1

log_error_prob=1-success/shots
print(log_error_prob)

#the majority voting is not neccesary since the corrected_state will be a logical state
#because logical_counter is 100% every time
# print(logical_counter)
# therefore, when using the pymatching decoder it is sufficient to look at a single bit
