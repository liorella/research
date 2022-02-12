import numpy as np
import stim
from stim import CircuitInstruction

from qec_generator import CircuitParams
from stim_lib.error_context import StimErrorContext
from stim_lib.scheduled_circuit import generate_scheduled, get_pauli_probs, instruction_duration
from stim_lib.run_feedback import to_measure_segments, do_and_get_measure_results


def gen_feedback_circuit(f_vec: np.ndarray,
                         qubits_in_reset: np.ndarray,
                         context: StimErrorContext):
    fc = stim.Circuit()
    params = context.params
    to_reset = qubits_in_reset[np.nonzero(f_vec)]
    fc.append_operation("X", to_reset)
    fc.append_operation('DEPOLARIZE1', to_reset, params.single_qubit_depolarization_rate)
    px, py, pz = get_pauli_probs(instruction_duration(CircuitInstruction('R', [0], [0]), params),
                                 params.t1,
                                 params.t2
                                 )
    fc.append_operation('PAULI_CHANNEL_1',
                        context.data_qubits,
                        [px, py, pz * (1 + params.meas_induced_dephasing_enhancement)]
                        )
    # print(fc)
    return fc


def experiment_run(circuit: stim.Circuit,
                   context: StimErrorContext,
                   shots: int,
                   reset_strategy: str) -> float:
    """
    generate a single experimental run with active parity tracking and return corrected logical state.
    Allows to choose the reset strategy.

    :param circuit: A circuit object, generated using `generate_scheduled()`
    :param context: A context object, generated using `generate_scheduled()`
    :param shots: The number of shots to perform
    :param reset_strategy: Either 'APT' or 'AR' for active parity tracking/active reset, respectively
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
        results_prev = np.zeros_like(f_vec)
        reset_indicator = np.zeros_like(f_vec)
        results_record = np.zeros((num_rounds + 1, len(context.active_ancillas)), dtype=np.uint8)
        sim = stim.TableauSimulator()

        for i, segment in enumerate(surf_circ_iter):
            if i < num_rounds:

                # inject error
                # if i == 1:
                #     ec = stim.Circuit()
                #     ec.append_operation('X', [10])
                #     sim.do(ec)
                #
                # if i == 2:
                #     ec = stim.Circuit()
                #     ec.append_operation('X', [14])
                #     sim.do(ec)

                sim.do(gen_feedback_circuit(reset_indicator, context.active_ancillas, context))
                results = do_and_get_measure_results(sim, segment, context.active_ancillas)
                results_record[i, :] = results
                if reset_strategy == 'APT':
                    f_vec = (f_vec + results_prev) % 2
                    results_prev = results
                    reset_indicator = f_vec
                elif reset_strategy == 'AR':
                    reset_indicator = results
                else:
                    raise ValueError(f'unknown reset strategy {reset_strategy}')
            elif i == num_rounds:
                results_data = do_and_get_measure_results(sim, segment, context.data_qubits)
                data_parity = context.matching_matrix @ results_data % 2
                if reset_strategy == 'APT':
                    # the following is because D_n = p_n + p_{n-1} and p_{n-1} = m_{n-1} + f_n
                    results = (f_vec + data_parity + results_prev) % 2
                    results_record[i, :] = results
                    to_decode = results_record.T
                elif reset_strategy == 'AR':
                    results_record[i, :] = data_parity
                    to_decode = np.diff(np.hstack((np.zeros_like(f_vec)[:, np.newaxis], results_record.T))) % 2
                else:
                    raise ValueError(f'unknown reset strategy {reset_strategy}')
                frame = context.decode(to_decode)
                # print('frame = ', frame)
                # print('flipped qubits = ', np.array(context.data_qubits)[np.flatnonzero(frame)])
                corrected_state = (results_data + frame) % 2
                logical_state = context.logical_vecs @ corrected_state.T % 2
                naive_logical = context.logical_vecs @ results_data % 2
                # print(f'logical state = {logical_state} naive logical = {naive_logical}')
                # print(to_decode)
                if logical_state == 0:
                    success += 1
    return success / shots


if __name__ == '__main__':
    cparams = CircuitParams(t1=15e3,
                            t2=10e3,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=100,
                            single_qubit_depolarization_rate=0.01,
                            two_qubit_depolarization_rate=0.01,
                            meas_duration=600,
                            reset_duration=300,
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

    print('success = ', experiment_run(circ, cont, shots=1, reset_strategy='APT'))
