from qiskit.circuit import Instruction
from qiskit.circuit.library import XGate

config_base = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                i + 1: {'offset': 0.0} for i in range(10)
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0}
            }
        }
    },

    'elements': {
        'd0': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'
            },
            'oscillator': 0,
        },
        'd1': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb2'
            },
            'oscillator': 1,
        },
        'u0': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'
            },
            'oscillator': 1,
        },
        'u1': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb2'
            },
            'oscillator': 0,
        },
        'm0': {
            'mixInputs': {
                'I': ('con1', 9),
                'Q': ('con1', 10),
                'lo_frequency': 0.0,
                'mixer': 'mxr_rr'
            },
            'oscillator': 2,
            'operations': {
                'test_pulse_1': 'readout_pulse_in',
            },
            'time_of_flight': 200,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 1),
            }

        },
        'm1': {
            'mixInputs': {
                'I': ('con1', 9),
                'Q': ('con1', 10),
                'lo_frequency': 0.0,
                'mixer': 'mxr_rr'
            },
            'oscillator': 3,
            'operations': {
                'test_pulse_1': 'readout_pulse_in',  # todo: this is hardcoded, fix
            },
            'time_of_flight': 200,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 1),
            }

        },

    },
    'pulses': {
        'readout_pulse_in': {
            'operation': 'measurement',
            'length': 100,
            'waveforms': {
                'I': 'meas_wf_i',
                'Q': 'meas_wf_q',
            },
            'integration_weights': {
                'integw': 'integW1_I',
            },
            'digital_marker': 'trig_wf0'
        },
    },

    'digital_waveforms': {
        'trig_wf0': {'samples': [(1, 100), (0, 0)]}
    },

    'waveforms': {
        'meas_wf_i': {
            'type': 'constant',
            'sample': 0.4
        },
        'meas_wf_q': {
            'type': 'constant',
            'sample': 0.0
        },
    },

    'integration_weights': {
        'integW1_I': {
            'cosine': [0.1] * 25,
            'sine': [0.0] * 25,
        },
    },

    'mixers': {
        'mxr_rr': [
            {'intermediate_frequency': 0.0, 'lo_frequency': 0.0, 'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
        'mxr_qb1': [
            {'intermediate_frequency': 0.0, 'lo_frequency': 0.0,
             'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
        'mxr_qb2': [
            {'intermediate_frequency': 0.0, 'lo_frequency': 0.0,
             'correction': [1.0, 0.0, 0.0, 1.0]},
        ],
    },

}

XGate

