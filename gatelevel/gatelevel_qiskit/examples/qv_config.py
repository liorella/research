num_qubits = 4

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

    'elements': {**{
        f'd{i}': {
            'mixInputs': {
                'I': ('con1', 2*i + 1),  # these assignments don't make sense but not important for the demo
                'Q': ('con1', 2*i + 2),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'  # these assignments don't make sense but not important for the demo
            },
            'intermediate_frequency': 0.0,
        } for i in range(num_qubits)},
        **{
        f'u{i}': {
            'mixInputs': {
                'I': ('con1', 2*i + 5),  # these assignments don't make sense but not important for the demo
                'Q': ('con1', 2*i + 6),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'  # these assignments don't make sense but not important for the demo
            },
            'intermediate_frequency': 0.0,
        } for i in range(num_qubits)},
        **{
        f'm{i}': {
            'mixInputs': {
                'I': ('con1', 9),
                'Q': ('con1', 10),
                'lo_frequency': 0.0,
                'mixer': 'mxr_rr'
            },
            'intermediate_frequency': 0.0,
            'operations': {
                'test_pulse_1': 'readout_pulse_in',
            },
            'time_of_flight': 200,
            'smearing': 0,
            'outputs': {
                'out1': ('con1', 1),
            }

        } for i in range(num_qubits)},

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