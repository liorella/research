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
    record = np.array(sim.current_measurement_record()[-segment.num_measurements:])
    assert segment[-1].name in ('M', 'MR')
    meas_targets = [t.value for t in segment[-1].targets_copy()]
    assert len(meas_targets) == len(record)
    return record[np.isin(meas_targets, qubits_to_return)].astype(np.uint8)


def gen_feedback_circuit(f_vec: np.ndarray,
                         qubits_in_reset: np.ndarray):
    fc = stim.Circuit()
    to_reset = qubits_in_reset[np.nonzero(f_vec)]
    fc.append_operation("X", to_reset)
    print(fc)
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
    print(surf_circ)
    surf_context = StimErrorContext(surf_circ, num_rounds)

    surf_circ_iter = to_measure_segments(surf_circ)

    f_vec = np.zeros(len(surf_context.active_ancillas), dtype=np.uint8)
    results_prev = np.zeros(len(surf_context.active_ancillas), dtype=np.uint8)

    results_record = np.zeros((num_rounds + 1, len(surf_context.active_ancillas)), dtype=np.uint8)
    sim = stim.TableauSimulator()

    for i, segment in enumerate(surf_circ_iter):
        if i < num_rounds:

            if i == 1:
                # inject error
                ec = stim.Circuit()
                ec.append_operation('X', [5, 15])
                sim.do(ec)

            sim.do(gen_feedback_circuit(f_vec, surf_context.active_ancillas))
            results = do_and_get_measure_results(sim, segment, surf_context.active_ancillas)
            results_record[i, :] = results
            f_vec = (f_vec + results_prev) % 2
            results_prev = results
        elif i == num_rounds:
            results_data = do_and_get_measure_results(sim, segment, surf_context.data_qubits)
            data_parity = surf_context.matching_matrix @ results_data % 2
            # the following is because D_n = p_n + p_{n-1} and p_{n-1} = m_{n-1} + f_n
            results = (f_vec + data_parity + results_prev) % 2
            results_record[i, :] = results
            frame = surf_context.decode(results_record.T)
            print('frame = ', frame)
            print('flipped qubits = ', np.array(surf_context.data_qubits)[np.flatnonzero(frame)])
            corrected_state = (results_data + frame) % 2
            logical_state = surf_context.logical_vecs @ corrected_state.T % 2
            naive_logical = surf_context.logical_vecs @ results_data % 2
            print(f'logical state = {logical_state} naive logical = {naive_logical}')
            print(results_record.T)
            return logical_state


if __name__ == '__main__':
    cparams = CircuitParams(t1=np.inf,
                            t2=np.inf,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=100,
                            single_qubit_depolarization_rate=0,
                            two_qubit_depolarization_rate=0,
                            meas_duration=600,
                            reset_duration=0,
                            reset_latency=40)

    experiment_shot('surface_code:rotated_memory_z',
                    3,
                    2,
                    cparams)
