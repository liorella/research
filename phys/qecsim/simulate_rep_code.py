import quantumsim  # https://gitlab.com/quantumsim/quantumsim, branch: stable/v0.2
import numpy as np
from pymatching import Matching

import logging

from qecsim.rep_code_generator import RepCodeGenerator
from qecsim.qec_generator import CircuitParams

log = logging.getLogger('qec')
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

if __name__ == "__main__":
    # run simulation
    ########
    plot = False
    distance = 2
    encoded_data = True
    num_iterations = 100
    cparams = CircuitParams(t1=10e5,
                            t2=10e3,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=40,
                            meas_duration=400,
                            reset_duration=250,
                            reset_latency=200)

    repc = RepCodeGenerator(distance=distance,
                            circuit_params=cparams
                            )

    stabilizer = repc.generate_stabilizer_round(plot=plot)

    # start cycle
    success_vector = []
    for n in range(num_iterations):
        state = quantumsim.sparsedm.SparseDM(repc.register_names)
        results = []
        if encoded_data:
            repc.generate_logical_x(plot=plot).apply_to(state)
        for i in range(3):
            stabilizer.apply_to(state)
            results.append([state.classical[cb] for cb in repc.cbit_names[1::2]])
            # apply active reset
            to_reset = []
            for q, cb in zip(repc.qubit_names[1::2], repc.cbit_names[1::2]):
                if state.classical[cb] == 1:
                    to_reset.append(q)
            repc.generate_active_reset(to_reset).apply_to(state)

        repc.generate_bitflip_error(["0"], plot=plot).apply_to(state)

        for i in range(1):
            stabilizer.apply_to(state)
            results.append([state.classical[cb] for cb in repc.cbit_names[1::2]])
            # apply active reset
            to_reset = []
            for q, cb in zip(repc.qubit_names[1::2], repc.cbit_names[1::2]):
                if state.classical[cb] == 1:
                    to_reset.append(q)
            repc.generate_active_reset(to_reset).apply_to(state)

        repc.generate_bitflip_error(["4"]).apply_to(state)

        for i in range(6):
            stabilizer.apply_to(state)
            results.append([state.classical[cb] for cb in repc.cbit_names[1::2]])
            # apply active reset
            to_reset = []
            for q, cb in zip(repc.qubit_names[1::2], repc.cbit_names[1::2]):
                if state.classical[cb] == 1:
                    to_reset.append(q)
            repc.generate_active_reset(to_reset).apply_to(state)

        repc.generate_measure_data(plot=plot).apply_to(state)
        data_meas = np.array([state.classical[cb] for cb in repc.cbit_names[::2]])

        # postprocessing
        results = np.array(results)
        detection_events = np.vstack([[0] * distance, np.logical_xor(results[1:], results[:-1])])
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
        log.info(f"success = {success}")
        success_vector.append(success)
    log.info(f"success rate = {np.count_nonzero(success_vector) / len(success_vector)}")
