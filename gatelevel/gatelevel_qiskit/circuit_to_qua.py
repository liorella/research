import itertools
from copy import copy, deepcopy
from pprint import pprint

from qiskit import schedule as build_schedule, QuantumCircuit
from qiskit.circuit import Barrier
from qiskit.pulse import Play, ShiftPhase, Schedule, Acquire, AcquireChannel, MemorySlot, MeasureChannel, \
    ControlChannel, DriveChannel
import numpy as np

# helper functions
from gatelevel_qiskit.lib import write_indent_line
from gatelevel_qiskit.pulse_backend import PulseBackend

_play_channels = {MeasureChannel, ControlChannel, DriveChannel}


class CircuitQuaTransformer:
    def __init__(self,
                 pulse_backend: PulseBackend,
                 config_base: dict,
                 circuit: QuantumCircuit):
        self._circuit = circuit
        self._channel_port_map = CircuitQuaTransformer._create_channel_port_map(config_base)
        self._channel_freq_map = CircuitQuaTransformer._create_chan_freq_map(config_base)
        self._config_base = config_base
        self._pulse_backend = pulse_backend
        self.config = self._to_config()
        self._pulse_backend.add_measure_pulses(config_base)
        self._schedule = self._circuit_to_schedule(circuit)

    @staticmethod
    def _create_channel_port_map(config_base):
        ch_port_map = {}
        for element in config_base["elements"]:
            ch_port_map[element + "_i"] = str(config_base["elements"][element]["mixInputs"]["I"][1])
            ch_port_map[element + "_q"] = str(config_base["elements"][element]["mixInputs"]["Q"][1])
        return ch_port_map

    @staticmethod
    def _create_chan_freq_map(config_base):
        ch_freq_map = {}
        for element in config_base["elements"]:
            if element[0] == "d":
                channel = DriveChannel(int(element[1]))
            elif element[0] == "m":
                channel = MeasureChannel(int(element[1]))
            elif element[0] == "u":
                channel = ControlChannel(int(element[1]))
            else:
                raise ValueError(f"unknown element to channel mapping for element {element}")
            ch_freq_map[channel] = config_base["elements"][element]["intermediate_frequency"]
        return ch_freq_map

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

    def to_waveforms(self, init_time=0.0):
        schedule = self._schedule
        inst_dict = {item: [] for item in self._channel_port_map.values()}
        accumulated_phases = {chan: 0.0 for chan in schedule.channels}

        inst_seq = schedule.instructions
        for inst_tuple in inst_seq:
            start_time, inst = inst_tuple
            if isinstance(inst, ShiftPhase):
                accumulated_phases[inst.channel] += inst.phase
            elif isinstance(inst, Play):
                phase_wrapped = (accumulated_phases[inst.channel] - np.pi) % (2 * np.pi) - np.pi
                if phase_wrapped == -np.pi:
                    phase_wrapped = np.pi  # hack to make compatible with simulator
                inst_pre_samples = {
                    'duration': float(inst.pulse.duration),
                    'frequency': self._channel_freq_map[inst.channel],
                    'phase': phase_wrapped,
                    'timestamp': float(start_time) + init_time,
                }
                inst_i = copy(inst_pre_samples)
                inst_i['samples'] = {'values': inst.pulse.samples.real.tolist(),
                                     'type': 'literal'}
                inst_i['name'] = inst.name + "_i"
                inst_q = copy(inst_pre_samples)
                inst_q['samples'] = {'values': inst.pulse.samples.imag.tolist(),
                                     'type': 'literal'}
                inst_q['name'] = inst.name + "_q"
                inst_dict[self._channel_to_port(inst.channel, "i")].append(inst_i)
                inst_dict[self._channel_to_port(inst.channel, "q")].append(inst_q)
            elif isinstance(inst, Acquire):
                pass
            else:
                raise ValueError(f"unknown instruction type {inst}")

        inst_dict = {key: value for key, value in inst_dict.items() if len(value) > 0}

        return inst_dict

    def to_qua(self):
        schedule = self._schedule
        ops_by_chan = CircuitQuaTransformer._ops_by_chan(schedule)
        num_measure = len([key for key in ops_by_chan.keys() if isinstance(key, MeasureChannel)])

        qua_str = write_indent_line("with program() as prog:")
        qua_str += write_indent_line(f"I = [None] * {num_measure}", 1)
        for i in range(num_measure):
            qua_str += write_indent_line(f"I[{i}] = declare(fixed)", 1)
        qua_str += write_indent_line(
            f"align(*{[chan.name for chan in schedule.channels if type(chan) in _play_channels]})",
            1)
        inst_seq = schedule.instructions

        current_time_map = {chan: 0 for chan in schedule.channels}

        for inst_tuple in inst_seq:
            start_time, inst = inst_tuple
            if isinstance(inst, ShiftPhase):
                qua_str += write_indent_line(f"frame_rotation({inst.phase}, '{inst.channel.name}')", 1)
            elif isinstance(inst, Play):
                if start_time > current_time_map[inst.channel]:
                    wait_time = (start_time - current_time_map[inst.channel])
                    qua_str += write_indent_line(f"wait({wait_time // 4},"
                                                 f"'{inst.channel.name}')", 1)
                else:
                    wait_time = 0
                current_time_map[inst.channel] += inst.duration + wait_time
                if isinstance(inst.channel, MeasureChannel):
                    qua_str += write_indent_line(f"measure('test_pulse_1', '{inst.channel.name}', None, "  # todo: this is hardcoded, fix
                                                 f"('integw', I[{inst.channel.index}]))", 1)
                    qua_str += write_indent_line(f"save(I[{inst.channel.index}], 'I{inst.channel.index}')", 1)
                else:
                    qua_str += write_indent_line(f"play('{inst.name}', '{inst.channel.name}')", 1)

            elif isinstance(inst, Barrier):
                pass
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
        ops_by_chan = CircuitQuaTransformer._ops_by_chan(schedule)
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
