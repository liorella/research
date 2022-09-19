import math

from optimus_sim import Cal, maintain, HiddenState
import logging

log = logging.getLogger("optimus")
log.setLevel(logging.INFO)

# build graph
rr_spec = Cal("rr spec", set())
qubit_spec = Cal("q spec", {rr_spec})
t1 = Cal("t1", {qubit_spec})
t2 = Cal("t2", {qubit_spec})
rb = Cal("rb", {t1, t2})

nodes = {rr_spec, qubit_spec, t1, t2, rb}


def test_ok_scenario():
    # TODO: finish this
    for n in nodes:
        n.last_measured = 2
        n.staleness = 3
        n.hidden_state = HiddenState.ok

    maintain(rb)


def test_uncalibrated_scenario():
    for n in nodes:
        n.last_measured = math.inf
        n.staleness = 3
        n.hidden_state = HiddenState.uncalibrated

    maintain(rb)
    print("")
    for n in nodes:
        print(n)
    for n in nodes:
        assert n.hidden_state == HiddenState.ok
        assert n.last_measured == 0


def test_q_spec_is_stale_no_drift():
    """
    only q spec is stale, refresh it in a scenario where all check_data passes
    (nothing drifted)
    """
    for n in nodes:
        n.last_measured = 2
        n.staleness = 3
        n.hidden_state = HiddenState.ok

    qubit_spec.staleness = 1

    maintain(rb)
    print("")
    for n in nodes:
        print(n)
    for n in nodes:
        assert n.hidden_state == HiddenState.ok
    for n in nodes.difference({rr_spec}):
        assert n.last_measured == 0
    assert rr_spec.last_measured == 2


def test_q_spec_is_stale_with_drift():
    raise NotImplementedError()


# TODO: add more tests