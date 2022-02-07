import numpy as np
import stim

from qecsim.qec_generator import CircuitParams
from qecsim.stim_error_context import StimErrorContext
from qecsim.stim_scheduled_circuit import generate_scheduled
from stim_run_feedback import to_measure_segments


def do_and_get_measure_results(sim: stim.TableauSimulator,
                               segment: stim.Circuit,
                               qubits_to_return: np.ndarray
                               ) -> np.ndarray:
    sim.do(segment)
    record = sim.current_measurement_record()[-segment.num_measurements:]
    meas_targets = [t.value for t in segment[-1].targets_copy]
    assert len(meas_targets) == len(record)
    return record[np.isin(meas_targets, qubits_to_return)]


def gen_feedback_circuit(f_vec: np.ndarray,
                         qubits_in_reset: np.ndarray):
    fc = stim.Circuit()
    to_reset = qubits_in_reset[np.nonzero(f_vec)]
    fc.append_operation("X", to_reset)
    return fc


def experiment_shot(code_task: str, distance: int, num_rounds: int, params: CircuitParams) -> np.ndarray:
    """
    generate a single experimental run and return corrected logical state

    :param code_task: The code task, as defined by the input to stim.Circuit.generated
    :param distance: The code distance, as defined by the input to stim.Circuit.generated
    :param num_rounds: The number of memory rounds, as defined by the input to stim.Circuit.generated
    :param params: circuit parameters
    :return: the simulated logical state for this shot
    """
    if distance % 2 != 1:
        raise ValueError(f"only odd distance circuits possible. Distance = {distance}")

    surf_circ = generate_scheduled(
        code_task,
        distance=distance,
        rounds=num_rounds,
        params=params
    )
    surf_context = StimErrorContext(surf_circ)

    surf_circ_iter = to_measure_segments(surf_circ)

    f_vec = np.zeros(len(surf_context.active_ancillas), dtype=np.uint8)
    results_prev = np.zeros(len(surf_context.active_ancillas), dtype=np.uint8)

    sim = stim.TableauSimulator()

    for i, segment in enumerate(surf_circ_iter):
        if i < num_rounds:
            results = do_and_get_measure_results(sim, segment, surf_context.active_ancillas)
            f_vec = (f_vec + results_prev) % 2
            results_prev = results
            sim.do(gen_feedback_circuit(f_vec, surf_context.active_ancillas))
        elif i == num_rounds:
            results_data = do_and_get_measure_results(sim, segment, surf_context.data_qubits)
            results = surf_context.matching_matrix @ results_data % 2
            f_vec = (f_vec + results) % 2
            # frame = surf_context.pymatch_obj.decode(f_vec)
            # corrected_state = (results_data + frame) % 2
            # logical_state = surf_context.logical_vecs @ corrected_state.T % 2
    # return logical_state


if __name__ == '__main__':
    cparams = CircuitParams(t1=15e3,
                            t2=19e3,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=100,
                            single_qubit_depolarization_rate=1.1e-3,
                            two_qubit_depolarization_rate=6.6e-3,
                            meas_duration=600,
                            reset_duration=0,
                            reset_latency=40)

    experiment_shot('surface_code:rotated_memory_z',
                    3,
                    10,
                    cparams)
