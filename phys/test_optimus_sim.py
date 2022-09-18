import math

from optimus_sim import Cal, State, maintain

rr_spec = Cal("rr spec", set())
qubit_spec = Cal("q spec", {rr_spec})
t1 = Cal("t1", {qubit_spec})
t2 = Cal("t2", {qubit_spec})
rb = Cal("rb", {t1, t2})

nodes = {rr_spec, qubit_spec, t1, t2, rb}


def test_uncalibrated_scenario():
    for n in nodes:
        n.state = State.out_of_spec
        n.last_checked_ago = math.inf
        n.staleness = 3

    maintain(rb)
    for n in nodes:
        print(n)

