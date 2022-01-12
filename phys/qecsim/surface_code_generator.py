from quantumsim import circuit

from qecsim.qec_generator import QECGenerator, CircuitParams


class SurfaceCodeGenerator(QECGenerator):
    def __init__(self, distance: int, circuit_params: CircuitParams):
        super().__init__(distance, circuit_params)
        self.cbits = [
            circuit.ClassicalBit("m" + str(i)) for i in range(2 * self.distance ** 2 - 1)
        ]
        self.qubits = [
            circuit.Qubit(str(i), t1=self.params.t1, t2=self.params.t2) for i in range(2 * self.distance ** 2 - 1)
        ]

    def generate_state_encoder(self, state, plot=False) -> circuit.Circuit:
        pass

    def generate_stabilizer_round(self, final_round=False, injected_errors=None, plot=False) -> circuit.Circuit:
        pass

    def generate_active_reset(self, qubit_names, plot=False) -> circuit.Circuit:
        pass

    def generate_measure_data(self, plot=False) -> circuit.Circuit:
        pass
