from typing import Optional

import numpy as np
from dataclasses import dataclass


@dataclass
class ErrorProbs:
    cnot_bitflip: float = 0.0
    measure_bitflip: float = 0.0
    time_over_rate_01: float = 0.0
    time_over_rate_10: float = 0.0


# todo: turn this into an abstract base class that will allow us to implement it (at least partially) in stim and cirq
class QECCirc:
    def __init__(self,
                 distance: int,
                 creg_size: int,
                 error_probs: Optional[ErrorProbs] = None
                 ):
        self.qreg = [0] * (4 * distance + 1)
        self.creg = [False] * creg_size
        self.error_probs = ErrorProbs() if error_probs is None else error_probs

    def _cnot(self, qc, qt):
        cnot_table = {
            (0, 0): (0, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 1),
            (1, 1): (1, 0)
        }

        result = cnot_table[(self.qreg[qc], self.qreg[qt])]
        self.qreg[qc], self.qreg[qt] = result
        # error model: flip either of the qubits after applying the gate
        # todo: make sure this error model is correct
        if np.random.rand(1) < self.error_probs.cnot_bitflip:
            self.qreg[qc] = 1 - self.qreg[qc]
        if np.random.rand(1) < self.error_probs.cnot_bitflip:
            self.qreg[qt] = 1 - self.qreg[qt]

    def _measure(self, q, c):
        self.creg[c] = bool(self.qreg[q]) if np.random.rand(1) < self.error_probs.measure_bitflip else not bool(
            self.qreg[q])

    def _cl_not(self, c):
        self.creg[c] = not self.creg[c]

    def _cl_xor(self, c1, c2, cr):
        self.creg[cr] = self.creg[c1] ^ self.creg[c2]

    def _wait(self, q):
        """
        flip qubit q with probability 1 - exp(-time_over_rate_01) if qubit is in 1 state and 1 - exp(-time_over_rate_10)
        if qubit is in state 0
        """
        if self.qreg[q] == 1:
            tor = self.error_probs.time_over_rate_01
        else:
            tor = self.error_probs.time_over_rate_10
        self.qreg[q] = self.qreg[q] if np.random.rand(1) < np.exp(-tor) else 1 - self.qreg[q]

    def add_moment(self, ops_list):
        used_qubits = np.zeros(len(self.qreg))
        for op in ops_list:
            if op[0] == "CNOT":
                self._cnot(op[1], op[2])
                used_qubits[op[1]] += 1
                used_qubits[op[2]] += 1
            if op[0] == "M":
                self._measure(op[1], op[2])
                used_qubits[op[1]] += 1
                used_qubits[op[2]] += 1
            if op[0] == "WAIT":
                self._wait(op[1])
                used_qubits[op[1]] += 1
            if op[0] == "CL_NOT":
                self._cl_not(op[1])
            if op[0] == "CL_XOR":
                self._cl_xor(op[1], op[2], op[3])
        if not np.all(used_qubits == 1):
            raise AttributeError("every qubit must be used in every moment exactly once")

