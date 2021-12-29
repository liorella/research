import importlib

# Import Qiskit classes
import qiskit
# Import the qv function
from qiskit.circuit.library import QuantumVolume

from gatelevel_qiskit.circuit_to_qua import CircuitQuaTransformer
from gatelevel_qiskit.examples.qv_config import config_base, num_qubits
from gatelevel_qiskit.lib import write_indent_line
from gatelevel_qiskit.simple_backend import simple_backend, qubit_coupling_map


class QVMaker(CircuitQuaTransformer):
    def __init__(self):  # todo: generate config for number of qubits instead of making it hard coded and imported
        qvc = QuantumVolume(num_qubits, depth=num_qubits, classical_permutation=False)

        qcvt = qiskit.compiler.transpile(qvc,
                                         basis_gates=['u1', 'sx', 'cx'],
                                         coupling_map=qubit_coupling_map,
                                         optimization_level=3)

        super().__init__(simple_backend, config_base, qcvt)

    def make_qv_macro(self):
        py_str = ""
        py_str += write_indent_line("from qm.qua import *", 0)
        py_str += write_indent_line("def macro(args):", 0)
        py_str += write_indent_line("align()", 1)
        py_str += self._create_qua_body(py_str)
        py_str += write_indent_line("align()", 1)

        # fl = tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir='.')
        filename = 'tttt'
        with open(filename + '.py', 'w') as fl:
            fl.writelines(py_str)
        importlib.invalidate_caches()
        # importname = fl.name.split('/')[-1].split('.')[0]
        importname = filename
        myfile = importlib.import_module(importname)
        # myfile.my_macro()
        return myfile.__dict__["macro"]


if __name__ == "__main__":
    qvm = QVMaker()
    macro = qvm.make_qv_macro()
