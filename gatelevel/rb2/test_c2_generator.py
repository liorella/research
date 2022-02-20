from timeit import timeit
import numpy as np
import pytest
from qiskit import Aer, execute

from lib.c2_generator import clifford_to_unitary, index_to_clifford, size_c2, unitary_to_index, c2_unitaries, \
    generate_clifford_truncations, _clifford_seq_to_qiskit_circ, is_phase


@pytest.mark.parametrize('num_rand', [pytest.param(1),
                                      pytest.param(1000, marks=pytest.mark.stress)])
def test_clifford_to_unitary(num_rand):
    # in principle we should also test it stabilizes a Pauli - not doing this now
    indices = np.random.randint(size_c2, size=num_rand)
    for index in indices:
        unitary = clifford_to_unitary(index_to_clifford(index))
        assert unitary.shape == (4, 4)
        assert unitary.dtype == np.complex
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), atol=1e-10)


@pytest.mark.parametrize('num_rand', [pytest.param(1),
                                      pytest.param(1000, marks=pytest.mark.stress)])
def test_unitary_to_index(num_rand):
    indices = np.random.randint(size_c2, size=num_rand)
    for index in indices:
        assert unitary_to_index(c2_unitaries[index]) == index


@pytest.mark.parametrize('num_rand', [pytest.param(1),
                                      pytest.param(1000, marks=pytest.mark.stress)])
def test_find_inverse(num_rand):
    for index in np.random.randint(size_c2, size=num_rand):
        unitary = clifford_to_unitary(index_to_clifford(index))
        inverse_unitary = unitary.conj().T
        inverse_index = unitary_to_index(inverse_unitary)
        inverse_unitary_from_index = clifford_to_unitary(index_to_clifford(inverse_index))
        prod_with_inverse = inverse_unitary_from_index @ unitary
        assert is_phase(prod_with_inverse)


def test_unitary_to_index_fails_on_random():
    for _ in range(10):
        randmat = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        with pytest.raises(AssertionError):
            unitary_to_index(randmat)


def test_unitary_to_index_time():
    print('\nindex to gate time = ',
          timeit('clifford_to_unitary(index_to_clifford(534))',
                 'from lib.c2_generator import clifford_to_unitary, index_to_clifford, size_c2, unitary_to_index',
                 number=10) / 10, 'seconds')
    print('\nunitary to index time = ',
          timeit('unitary_to_index(c2_unitaries[534])',
                 'from lib.c2_generator import c2_unitaries, unitary_to_index',
                 number=100) / 100, 'seconds')


def test_generate_clifford_truncations_basic():
    truncations = generate_clifford_truncations(50, seed=1)
    for trunc in truncations:
        print('\n', trunc)
        print('\n')
        seq_circ = _clifford_seq_to_qiskit_circ(trunc)
        print(seq_circ)

        simulator = Aer.get_backend("unitary_simulator")
        unitary: np.ndarray = execute(seq_circ, simulator) \
            .result().get_unitary()

        print(np.around(unitary, 3))

        assert is_phase(unitary)


@pytest.mark.stress
@pytest.mark.parametrize('run', range(20))
def test_generate_clifford_truncations_stress(run):
    truncations = generate_clifford_truncations(500, seed=1)
    for trunc in truncations:
        print('\n', trunc)
        print('\n')
        seq_circ = _clifford_seq_to_qiskit_circ(trunc)
        print(seq_circ)

        simulator = Aer.get_backend("unitary_simulator")
        unitary: np.ndarray = execute(seq_circ, simulator) \
            .result().get_unitary()

        print(np.around(unitary, 3))

        assert is_phase(unitary)


