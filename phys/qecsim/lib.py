from quantumsim.sparsedm import SparseDM
from qutip import fock_dm, Qobj, tensor


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
