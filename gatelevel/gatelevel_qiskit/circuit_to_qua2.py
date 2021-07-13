import itertools
from copy import copy, deepcopy
from pprint import pprint

from qiskit import schedule as build_schedule, QuantumCircuit
from qiskit.circuit import Barrier
from qiskit.pulse import Play, ShiftPhase, Schedule, Acquire, AcquireChannel, MemorySlot, MeasureChannel, \
    ControlChannel, DriveChannel
import numpy as np

# helper functions
from qiskit.pulse.channels import PulseChannel

from gatelevel_qiskit.lib import write_indent_line
from gatelevel_qiskit.pulse_backend import PulseBackend


class CircuitQua2Transformer:
    def __init__(self,
                 pulse_backend: PulseBackend,
                 config_base: dict,
                 circuit: QuantumCircuit):
        self._circuit = circuit
        self._channel_port_map = CircuitQua2Transformer._create_channel_port_map(config_base)
        self._channel_osc_map = CircuitQua2Transformer._create_chan_freq_map(config_base)
        self._config_base = config_base
        self._pulse_backend = pulse_backend
        self.config = self._to_config()
        self._pulse_backend.add_measure_pulses(config_base)
        self._schedule = self._circuit_to_schedule(circuit)

    @staticmethod
    def _create_channel_port_map(config_base):
        ch_port_map = {}
        for element in config_base["elements"]:
            ch_port_map[element + "_i"] = config_base["elements"][element]["mixInputs"]["I"][1]
            ch_port_map[element + "_q"] = config_base["elements"][element]["mixInputs"]["Q"][1]
        return ch_port_map

    @staticmethod
    def _create_chan_freq_map(config_base):
        ch_osc_map = {}
        for element in config_base["elements"]:
            if element[0] == "d":
                channel = DriveChannel(int(element[1]))
            elif element[0] == "m":
                channel = MeasureChannel(int(element[1]))
            elif element[0] == "u":
                channel = ControlChannel(int(element[1]))
            else:
                raise ValueError(f"unknown element to channel mapping for element {element}")
            ch_osc_map[channel] = config_base["elements"][element]["oscillator"]
        return ch_osc_map

    def _circuit_to_schedule(self, circuit: QuantumCircuit):
        return build_schedule(circuit, self._pulse_backend.backend,
                              inst_map=self._pulse_backend.instruction_schedule_map)

    def _channel_to_port(self, channel, quad):
        return self._channel_port_map[channel.name + "_" + quad]

    @staticmethod
    def _ops_by_chan(schedule: Schedule):
        chans_pulse_dict = {channel: set() for channel in schedule.channels}
        for inst in schedule.instructions:
            if isinstance(inst[1], Play):
                chans_pulse_dict[inst[1].channel].add(inst[1].name)
        return chans_pulse_dict

    def to_qua(self):
        schedule = self._schedule
        ops_by_chan = CircuitQua2Transformer._ops_by_chan(schedule)
        num_measure = len([key for key in ops_by_chan.keys() if isinstance(key, MeasureChannel)])

        channel_schedules = [schedule.filter(channels=[chan]) for chan in schedule.channels if
                             isinstance(chan, PulseChannel)]

        qua_str = ""
        qua_str += write_indent_line("deterministic:")
        qua_str = write_indent_line("no_gaps:", 1)
        qua_str += write_indent_line("parallel:", 2)
        for chan_index, channel_schedule in enumerate(channel_schedules):
            qua_str += write_indent_line(f"do('{channel_schedule.channels[0].name}'):", 3)
            inst_seq = channel_schedule.instructions

            current_time = 0

            for inst_tuple in inst_seq:
                start_time, inst = inst_tuple
                if isinstance(inst, ShiftPhase):
                    qua_str += write_indent_line(f"frame_rotation_2pi(phase={inst.phase / (2 * np.pi)},"
                                                 f"frame={chan_index})", 4)
                elif isinstance(inst, Play):
                    if start_time > current_time:
                        wait_time = start_time - current_time
                        qua_str += write_indent_line(f"wait({wait_time // 4})", 4)
                    else:
                        wait_time = 0
                    current_time += inst.duration + wait_time
                    ports = [f"(con1, {self._channel_port_map[inst.channel.name + '_' + quad]})" for quad in 'iq']
                    if isinstance(inst.channel, MeasureChannel):
                        qua_str += write_indent_line(
                            f"measure('test_pulse_1', "
                            f"oscillator={self._channel_osc_map[inst.channel]},"
                            f"frame={chan_index}, "
                            f"port={ports}, "  # todo: this is hardcoded, fix
                            f"('integw', I[{inst.channel.index}]))", 4)
                    else:
                        qua_str += write_indent_line(f"play('{inst.name}', "
                                                     f"oscillator={self._channel_osc_map[inst.channel]}, "
                                                     f"frame={chan_index}, "
                                                     f"port={ports})", 4)

                elif isinstance(inst, Barrier):
                    pass  # this will probably require going outside of strict timing
                elif isinstance(inst, Acquire):
                    pass
                else:
                    raise ValueError(f"unknown instruction type {inst}")
        return qua_str

    def _to_config(self):
        schedule = self._circuit_to_schedule(self._circuit)
        schedule = schedule.filter(channels=[DriveChannel(0), DriveChannel(1), ControlChannel(0), ControlChannel(1)])
        pulse_dict = {inst[1].name:
                          (inst[1].duration,
                           inst[1].pulse.samples)
                      for inst in schedule.instructions if isinstance(inst[1], Play)
                      }
        config = deepcopy(self._config_base)
        ops_by_chan = CircuitQua2Transformer._ops_by_chan(schedule)
        for chan in schedule.channels:
            if isinstance(chan, DriveChannel) or isinstance(chan, ControlChannel):
                config['elements'][chan.name]['operations'] = {value: value + '_in' for value in ops_by_chan[chan]}
        config['pulses'].update({pulse_name + '_in': {
            'operation': 'control',
            'length': pulse_dict[pulse_name][0],
            'waveforms': {
                'I': pulse_name + '_i',
                'Q': pulse_name + '_q',

            }

        } for pulse_name in pulse_dict.keys()})

        wf_names = set(p + "_i" for p in pulse_dict.keys()).union(set(p + "_q" for p in pulse_dict.keys()))

        config['waveforms'].update({wf_name: {
            'type': 'arbitrary',
            'samples': pulse_dict[wf_name[:-2]][1].real.tolist()
            if wf_name[-1] == 'i'
            else
            pulse_dict[wf_name[:-2]][1].imag.tolist()
        } for wf_name in wf_names})
        return config

    def get_qua_prog_obj(self, circuit: QuantumCircuit):
        # todo: not working, probably globals is wrong
        qua_str = self.to_qua(circuit)

        ret_dict = {}
        exec(qua_str, globals(), ret_dict)
        return ret_dict["prog"]
