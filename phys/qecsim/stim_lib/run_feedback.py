from typing import Iterator

import numpy as np
import stim

measure_instructions = ('M', 'MR', 'MX', 'MRX', 'MRY', 'MRZ', 'MX', 'MY', 'MZ')


def to_measure_segments(circuit: stim.Circuit) -> Iterator[stim.Circuit]:
    """
    Create an iterator that iterates over a stim_lib program and returns circuit segments that terminate
    in a measure-like instruction, or the last segment, taking control flow into account.

    This is useful when we want to feedback on measurement results.

    :param circuit: The circuit to transform
    :return: None
    """

    circ_segment = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            for _ in range(inst.repeat_count):
                for circ_segment2 in to_measure_segments(inst.body_copy()):
                    circ_segment += circ_segment2
                    if circ_segment[-1].name in measure_instructions:
                        yield circ_segment
                        circ_segment = stim.Circuit()
        elif inst.name not in measure_instructions:
            circ_segment.append_operation(inst)
        else:
            circ_segment.append_operation(inst)
            yield circ_segment
            circ_segment = stim.Circuit()
    if len(circ_segment) > 0:
        yield circ_segment


def do_and_get_measure_results(sim: stim.TableauSimulator,
                               segment: stim.Circuit,
                               qubits_to_return: np.ndarray
                               ) -> np.ndarray:
    sim.do(segment)
    record = np.array(sim.current_measurement_record()[-segment.num_measurements:])
    assert segment[-1].name in measure_instructions, f'bug - segment name is {segment[-1].name}'
    meas_targets = [t.value for t in segment[-1].targets_copy()]
    assert len(meas_targets) == len(record)
    return record[np.isin(meas_targets, qubits_to_return)].astype(np.uint8)


if __name__ == '__main__':

    genc = stim.Circuit.generated('surface_code:rotated_memory_z',
                                  distance=3,
                                  rounds=4,
                                  )
    gen = to_measure_segments(genc)
    while True:
        try:
            print(next(gen))
            print('-' * 30)
        except StopIteration:
            break
