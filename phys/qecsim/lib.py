import logging
from glob import glob

from quantumsim.sparsedm import SparseDM
from qutip import fock_dm, Qobj, tensor
import os
import numpy as np
from qecsim.rep_code_generator import CircuitParams

from vlogging import VisualRecord


class SimDataHandler:
    def __init__(self):
        self._counter = _get_save_counter()
        self.log = logging.getLogger('qec')
        self.log.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        self._save_dir = None

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

    def init_save_folder(self, name=None):
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self._save_dir = _mk_save_dir(name)

        fh = logging.FileHandler(os.path.join('data/', 'log.html'))
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

    def save_data(self, data_to_save: dict):
        np.savez(os.path.join(self._save_dir, 'data.npz'), **data_to_save)

    def save_params(self, params: CircuitParams):
        self.log.info(repr(params) + '\n')
        with open(os.path.join(self._save_dir, 'params.json'), 'w') as fl:
            fl.write(params.to_json())


def _get_save_counter():
    counter = 0
    if not os.path.isdir('data'):
        os.mkdir('data')
        with open('data/counter', 'w') as fl:
            fl.write(str(counter))
    elif not os.path.isfile('data/counter'):
        with open('data/counter', 'w') as fl:
            fl.write(str(counter))
    else:
        with open('data/counter', 'r') as fl:
            counter = int(fl.readline())
    return counter


def _mk_save_dir(name=None):
    if name is None:
        name = 'run'
    counter = _get_save_counter()
    savedir = os.path.join('data', f'run_{name}_{counter}')
    counter += 1
    with open(os.path.join('data', 'counter'), 'w') as fl:
        fl.write(str(counter))
    os.mkdir(savedir)
    return savedir


def _get_save_dir(counter):
    names = glob('data/run_*')
    counter_values = [int(name.split('_')[-1]) for name in names]
    return names[counter_values.index(counter)]


def _get_max_counter():
    names = glob('data/run_*')
    counter_values = [int(name.split('_')[-1]) for name in names]
    return max(counter_values)


def load_run_data(counter):
    dirname = _get_save_dir(counter)
    return np.load(os.path.join(dirname, 'data.npz'))


def quantumsim_dm_to_qutip_dm(state: SparseDM) -> Qobj:
    """
    Convert a quantumsim density matrix to a Qutip density matrix.

    Warning: Only 2-level qubits are supported
    :param state:
    :return:
    """
    dummy_dm = tensor([fock_dm(2) for _ in range(len(state.idx_in_full_dm))])
    return Qobj(inpt=state.full_dm.to_array(),
                dims=dummy_dm.dims,
                shape=dummy_dm.shape,
                type=dummy_dm.type,
                isherm=dummy_dm.isherm,
                isunitary=dummy_dm.isunitary)


if __name__ == "__main__":
    os.chdir('..')
    sdh = SimDataHandler()
    sdh.init_save_folder('fff')
    import matplotlib.pyplot as plt

    cparams = CircuitParams(t1=15e3,
                            t2=15e3,
                            single_qubit_gate_duration=20,
                            two_qubit_gate_duration=40,
                            meas_duration=400,
                            reset_duration=250,
                            reset_latency=0)
    f1 = plt.figure()
    plt.plot([1, 2, 3])
    sdh.log.info(VisualRecord('an image', f1, 'notes'))
    sdh.log.info('something\n')
    sdh.save_data({'a': np.array([1, 2, 3])})
    sdh.save_params(cparams)
    print(load_run_data(sdh._counter)['a'])
