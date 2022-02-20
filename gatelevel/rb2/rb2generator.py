from collections import UserDict
from dataclasses import dataclass

from lib.gateconcatenator import GateConcatenator, Moment
from lib.c2_generator import generate_clifford_truncations


class RB2Generator:
    pass


class MomentMap(UserDict):
    def __init__(self, data):
        super().__init__(data)
        mandatory_keys = {
            'I',
            'X',
            'Y',
            'X/2',
            'Y/2',
            '-X/2',
            '-Y/2',
            'cz'
        }
        for key in mandatory_keys:
            if key not in self.data.keys():
                raise ValueError('...')


clifford_sequence = generate_clifford_truncations(10, [5, 7], 0)


def cseq_to_gate_seq(clifford_seq, moment_map: MomentMap):
    moment_seq = []
    for clifford in clifford_seq:
        for q0g, q1g in zip(*clifford):
            if q0g == 'cz':
                moment_seq.append()


rb_config = GateConcatenator()
