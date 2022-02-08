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
    assert segment[-1].name in ('M', 'MR', 'MX', 'MRX', 'MRY', 'MRZ', 'MX', 'MY', 'MZ'), \
        f'bug - segment name is {segment[-1].name}'
    meas_targets = [t.value for t in segment[-1].targets_copy()]
    assert len(meas_targets) == len(record)
    return record[np.isin(meas_targets, qubits_to_return)].astype(np.uint8)


def gen_feedback_circuit(f_vec: np.ndarray,
                         qubits_in_reset: np.ndarray):
    fc = stim.Circuit()
    to_reset = qubits_in_reset[np.nonzero(f_vec)]
    fc.append_operation("X", to_reset)
    return fc


# todo: the experiment shot should handle correctly the reception of these arguments
def experiment_run_apt(circuit: stim.Circuit,
                       context: StimErrorContext,
                       shots: int) -> float:
    """
    generate a single experimental run with active parity tracking and return corrected logical state

    :param circuit: A circuit object, generated using `generate_scheduled()`
    :param context: A context object, generated using `generate_scheduled()`
    :param shots: The number of shots to perform
    :return: The success rate, which is the number of times the recovered logical state was 0
    """

    distance = context.distance
    num_rounds = context.rounds
    if distance % 2 != 1:
        raise ValueError(f"only odd distance circuits possible. Distance = {distance}")

    success = 0
    for shot in range(shots):
        surf_circ_iter = to_measure_segments(circuit)

        f_vec = np.zeros(len(context.active_ancillas), dtype=np.uint8)
        results_prev = np.zeros(len(context.active_ancillas), dtype=np.uint8)

        results_record = np.zeros((num_rounds + 1, len(context.active_ancillas)), dtype=np.uint8)
        sim = stim.TableauSimulator()

        for i, segment in enumerate(surf_circ_iter):
            if i < num_rounds:

                if i == 1:
                    # inject error
                    ec = stim.Circuit()
                    ec.append_operation('X', [14])
                    sim.do(ec)

                sim.do(gen_feedback_circuit(f_vec, context.active_ancillas))
                results = do_and_get_measure_results(sim, segment, context.active_ancillas)
                results_record[i, :] = results
                f_vec = (f_vec + results_prev) % 2
                results_prev = results
            elif i == num_rounds:
                results_data = do_and_get_measure_results(sim, segment, context.data_qubits)
                data_parity = context.matching_matrix @ results_data % 2
                # the following is because D_n = p_n + p_{n-1} and p_{n-1} = m_{n-1} + f_n
                results = (f_vec + data_parity + results_prev) % 2
                results_record[i, :] = results
                frame = context.decode(results_record.T)
                print('frame = ', frame)
                print('flipped qubits = ', np.array(context.data_qubits)[np.flatnonzero(frame)])
                corrected_state = (results_data + frame) % 2
                logical_state = context.logical_vecs @ corrected_state.T % 2
                naive_logical = context.logical_vecs @ results_data % 2
                print(f'logical state = {logical_state} naive logical = {naive_logical}')
                print(results_record.T)
                if logical_state == 0:
                    success += 1
    return success / shots


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

    # task = 'repetition_code:memory'  # looks ok
    # task = 'surface_code:rotated_memory_x'  # todo: doesn't run
    task = 'surface_code:rotated_memory_z'  # looks ok
    # task = 'surface_code:unrotated_memory_z'  # looks ok

    circ, cont = generate_scheduled(
        task,
        distance=3,
        rounds=20,
        params=cparams
    )
    print(circ)

    print('success = ', experiment_run_apt(circ, cont, shots=10))
