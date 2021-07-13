from typing import Union
from typing import Iterable

from qiskit.providers import Backend, BaseBackend
from qiskit.pulse import InstructionScheduleMap, Schedule, MeasureChannel, MemorySlot, AcquireChannel, Acquire, Play, \
    Constant

from qiskit import schedule as build_schedule, QuantumCircuit


class PulseBackend:
    def __init__(self, backend: BaseBackend, instruction_schedule_map: InstructionScheduleMap):
        self.backend = backend
        self.instruction_schedule_map = instruction_schedule_map

    def add(self,
            instruction: str,
            qubits: Union[int, Iterable[int]],
            schedule: Schedule) -> None:
        self.instruction_schedule_map.add(instruction, qubits, schedule)

    def add_measure_pulses(self, config_base):
        # todo: currently only constant measurement pulses supported, add arbitrarty
        # todo: support more than one measure pulse
        for meas_tuple in self.backend.configuration().meas_map:
            meas_sched = Schedule(name="m" + str(meas_tuple))
            for element in config_base["elements"]:
                if element[0] == "m":  # measure elements
                    channel_index = int(element[1])
                    if channel_index in meas_tuple:
                        operations = list(config_base["elements"][element]["operations"].keys())
                        if len(operations) > 1:
                            raise ValueError("only 1 meas pulse is supported")
                        else:
                            operation = operations[0]
                        pulse_name = config_base["elements"][element]["operations"][operation]
                        i_waveform_name = config_base["pulses"][pulse_name]["waveforms"]["I"]
                        q_waveform_name = config_base["pulses"][pulse_name]["waveforms"]["Q"]
                        waveform_amp = config_base["waveforms"][i_waveform_name]["sample"] + \
                                       1j * config_base["waveforms"][q_waveform_name]["sample"]
                        waveform_duration = config_base["pulses"][pulse_name]["length"]
                        meas_sched += Play(Constant(duration=waveform_duration,
                                                    amp=waveform_amp).get_waveform(),
                                           MeasureChannel(channel_index),
                                           "meas_wf")
                        meas_sched += Acquire(waveform_duration, AcquireChannel(channel_index), MemorySlot(channel_index))
            self.add("measure", meas_tuple, meas_sched)

    def build_schedule(self, circuit):
        return build_schedule(circuit, self.backend,
                              inst_map=self.instruction_schedule_map)
