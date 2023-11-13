spi_clk_speed = 10e6
spi_clk_period = max(int(1 / spi_clk_speed), 100)  # in ns
transaction_size_bits = 4
cs_period = spi_clk_period * (transaction_size_bits)
clk_cycle = [(0, spi_clk_period / 4), (1, spi_clk_period / 2), (0, spi_clk_period / 4)]

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": 0.0}
            },
            "analog_inputs": {
                1: {"offset": 0.0}
            },
            "digital_outputs": {3: {'inverted': True},
                                4: {},
                                5: {},
                                9: {},
                                10: {}},
        }
    },
    "elements": {
        "CS_n": {
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "clear": "clear_cs"
            },
        },
        "SCLK": {
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 5),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "tick": "raise_clock",
            },
        },
        "MOSI": {
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 4),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "send_bit": "start_data_transfer",
                "wait_cycles": "wait_cycle",
                "wait_four_cycles": "wait_four_cycle",
            },
        },
        "MEAS_IN": {
            "singleInput": {
                'port': ('con1', 1),
            },
            'intermediate_frequency': 0,
            'operations': {
                'readout': 'readout_pulse'
            },
            "outputs": {
                "out1": ("con1", 1)
            },
            "time_of_flight": 24,
            "smearing": 0
        },
        "TOTRIG": {
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 9),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "to_trig": "clear_cs",
            },
        },
        "TOMESURE": {
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 10),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "send_bit": "start_data_transfer",
            },
        },
    },
    "pulses": {
        "start_data_transfer": {
            "operation": "control",
            "length": spi_clk_period,
            "digital_marker": "set_high",
        },
        "raise_clock": {
            "operation": "control",
            "length": spi_clk_period * transaction_size_bits,
            "digital_marker": "clk",
        },
        "clear_cs": {
            "operation": "control",
            "length": cs_period,
            "digital_marker": "cs_n",
        },
        "wait_cycle": {
            "operation": "control",
            "length": spi_clk_period,
            "digital_marker": "set_low",
        },
        "wait_four_cycle": {
            "operation": "control",
            "length": spi_clk_period*4,
            "digital_marker": "set_four_low",
        },

        "readout_pulse": {
            "operation": "measurement",
            "length": 200,
            "waveforms": {
                "single": "zero_wf"
            },
            "integration_weights": {
                "const": "const"
            },
            "digital_marker": "ON"
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "set_low": {"samples": [(0, spi_clk_period)]},
        "set_four_low": {"samples": [(0, 4*spi_clk_period)]},
        "set_high": {"samples": [(1, spi_clk_period)]},
        "clk": {"samples": clk_cycle*transaction_size_bits},
        "to_trig": {"samples": [(1,0)]},
        "cs_n": {"samples": [(1, cs_period)],}  # note that this is inverted
    },
    "waveforms": {
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        }
    },
    "integration_weights": {
        "const": {
            "cosine": [(1.0, 200)],
            "sine": [(0.0, 200)],
        }
    },
}
