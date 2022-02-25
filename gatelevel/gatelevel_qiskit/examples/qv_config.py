num_qubits = 2

assert num_qubits <= 5, "hardcoded to have only 3 controllers"

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
        },
        'con2': {
            'type': 'opx1',
            'analog_outputs': {
                i + 1: {'offset': 0.0} for i in range(10)
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0}
            }
        },
        'con3': {
            'type': 'opx1',
            'analog_outputs': {
                i + 1: {'offset': 0.0} for i in range(10)
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0}
            }
        },
    },

    'elements': {**{
        f'd{i}': {
            'mixInputs': {
                'I': ('con1', 2 * i + 1),  # these assignments don't make sense but not important for the demo
                'Q': ('con1', 2 * i + 2),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'  # these assignments don't make sense but not important for the demo
            },
            'intermediate_frequency': 0.0,
        }
        for i in range(4)},
        'd4': {
            'mixInputs': {
                'I': ('con3', 3),  # these assignments don't make sense but not important for the demo
                'Q': ('con3', 4),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'  # these assignments don't make sense but not important for the demo
            },
            'intermediate_frequency': 0.0,
        },
        **{
            f'u{i}': {
                'mixInputs': {
                    'I': ('con2', 2 * i + 1),  # these assignments don't make sense but not important for the demo
                    'Q': ('con2', 2 * i + 2),
                    'lo_frequency': 0.0,
                    'mixer': 'mxr_qb1'  # these assignments don't make sense but not important for the demo
                },
                'intermediate_frequency': 0.0,
            } for i in range(4)},
        'u4': {
            'mixInputs': {
                'I': ('con3', 1),  # these assignments don't make sense but not important for the demo
                'Q': ('con3', 2),
                'lo_frequency': 0.0,
                'mixer': 'mxr_qb1'  # these assignments don't make sense but not important for the demo
            },
            'intermediate_frequency': 0.0,
        },
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
                    'readout_pulse': 'readout_pulse_in',
                },
                'time_of_flight': 200,
                'smearing': 0,
                'outputs': {
                    'out1': ('con1', 1),
                    'out2': ('con1', 2),
                }

            } for i in range(4)},

        f'm4': {
            'mixInputs': {
                'I': ('con3', 9),
                'Q': ('con3', 10),
                'lo_frequency': 0.0,
                'mixer': 'mxr_rr'
            },
            'intermediate_frequency': 0.0,
            'operations': {
                'readout_pulse': 'readout_pulse_in',
            },
            'time_of_flight': 360,
            'smearing': 0,
            'outputs': {
                'out1': ('con3', 1),
                'out2': ('con3', 2),
            }

        },
    },
    'pulses': {
        'readout_pulse_in': {
            'operation': 'measurement',
            'length':   500,
            'waveforms': {
                'I': 'meas_wf_i',
                'Q': 'meas_wf_q',
            },
            'integration_weights': {
                'iw1': 'iw1',
                'iw2': 'iw2',
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
        'iw1': {
            'cosine': [0.1] * (500 // 4),
            'sine': [0.0] * (500 // 4),
        },
        'iw2': {
            'cosine': [0.0] * (500 // 4),
            'sine': [0.1] * (500 // 4),
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
