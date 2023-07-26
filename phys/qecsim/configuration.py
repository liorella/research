spi_clk_speed = 50e6
spi_clk_period = max(int(1 / spi_clk_speed), 20)  # in ns
transaction_size_bits = 4
cs_period = spi_clk_period * (transaction_size_bits)
clk_cycle = [(0, spi_clk_period / 4), (1, spi_clk_period / 2), (0, spi_clk_period / 4)]

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
            },
            "digital_outputs": {1: {'inverted': True},
                                2: {},
                                3: {}},
        }
    },
    "elements": {
        "CS_n": {
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
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
                    "port": ("con1", 2),
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
                    "port": ("con1", 3),
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
    },
    "digital_waveforms": {
        "set_high": {"samples": [(1, spi_clk_period)]},
        "clk": {"samples": clk_cycle*transaction_size_bits},
        "cs_n": {"samples": [(1, cs_period)]}  # note that this is inverted
    },
}
