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
rounds=2
distance=5
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
                code_task='repetition_code:memory',  # looks ok
                distance=distance,
                rounds=rounds,
                t1_t2_depolarization=False,
                meas_induced_dephasing_enhancement=False,
                params=cparams
                ) # does not include before_round_data_depolarization=p and before_measure_flip_probability=p
# The context include the matching matrix

## running an experiment can be done through the function 'experiment_run'
shots=1000
reset_strategy='AR'
experiment_run(circuit, context, shots=shots, reset_strategy=reset_strategy)

## breaking down the 'experiment_run' (only active reset, and nodify to majority voting)
num_rounds = context.rounds

success = 0
for shot in range(shots):
    rep_circ_iter = to_measure_segments(circuit) #breaking down the circuit into segments, divided by measurement events
    reset_indicator = np.zeros(len(context.active_ancillas), dtype=np.uint8)

    results_record = np.zeros((num_rounds + 1, len(context.active_ancillas)), dtype=np.uint8) # a matrix that will eventually reach the decoder
    sim = stim.TableauSimulator() #neccesary to add components into the circuit

    for i, segment in enumerate(rep_circ_iter):
        if i < num_rounds:
            sim.do(gen_feedback_circuit(reset_indicator, context.active_ancillas, context)) # gives an X gate for the ancilla the is to reset and append errors
            results = do_and_get_measure_results(sim, segment, context.active_ancillas)  # simulate the segment
            results_record[i, :] = results
            reset_indicator = results
#            print(i)
 #           print(results)
        elif i == num_rounds:
            results_data = do_and_get_measure_results(sim, segment, context.data_qubits)
            data_parity = context.matching_matrix @ results_data % 2
            results_record[i, :] = data_parity
            to_decode = np.diff(np.hstack((np.zeros(len(context.active_ancillas), dtype=np.uint8)[:, np.newaxis], results_record.T))) % 2
            frame = context.decode(to_decode)
            corrected_state = (results_data + frame) % 2
            logical_state = np.around(np.sum(corrected_state)/context.distance, 0)
            if logical_state == 0:
                success += 1

log_error_prob=1-success/shots
print(log_error_prob)