# simulate optimus traversal, based on https://arxiv.org/pdf/1803.03226.pdf

# graph
import logging
from enum import Enum

log = logging.getLogger("optimus")
log.addHandler(logging.StreamHandler())


class HiddenState(Enum):
    uncalibrated = -1
    ok = 0
    drifted = 1  # calibration needed on this node only
    corrupted = 2  # bad data due to some downstream node being drifted
    failure = 3  # bad data, uncorrectable


class Cal:
    def __init__(self, name, dependencies):
        self.name = name
        self.last_measured = None
        self.staleness = None
        self.dependencies = dependencies
        self.hidden_state = HiddenState.uncalibrated

    def __repr__(self):
        return f"<name: {self.name} | state: {self.hidden_state} | " \
               f"last measured: {self.last_measured} | staleness: {self.staleness}>"


class Result:
    def __init__(self):
        self.success = None
        self.in_spec = None
        self.bad_data = None


def check_state(node: Cal, mode='soft') -> Result:
    log.info(f"checking state for {node}")
    res = Result()
    stale = node.last_measured > node.staleness
    recalibrated_dependencies = any(d.last_measured < node.last_measured
                                    for d in node.dependencies)
    if mode == 'soft':
        failing_dependencies = False
    elif mode == 'strict':
        failing_dependencies = any(not check_state(d).success for d in node.dependencies)  # recursive
    else:
        raise ValueError(f"unknown mode {mode}")
    res.success = not (stale or recalibrated_dependencies or failing_dependencies)
    return res


def check_data(node: Cal) -> Result:
    log.info(f"checking data for {node}")
    res = Result()
    if node.hidden_state == HiddenState.ok:
        res.in_spec = True
        res.bad_data = False
    elif node.hidden_state == HiddenState.drifted or node.hidden_state == HiddenState.uncalibrated:
        res.in_spec = False
        res.bad_data = False
    elif node.hidden_state == HiddenState.corrupted or node.hidden_state == HiddenState.failure:
        res.in_spec = False
        res.bad_data = True
    node.last_measured = 0
    return res


class CalibrationError(Exception):
    pass


def calibrate(node: Cal) -> Result:
    log.info(f"calibrating {node}")
    res = Result()
    if node.hidden_state.value < 3:  # succeed to calibrate in all cases except failure
        node.hidden_state = HiddenState.ok
    else:
        raise CalibrationError(f"calibration for node {node} failed")
    node.last_measured = 0
    return res


def update_parameters(result: Result) -> None:
    pass


def diagnose(node: Cal) -> bool:
    result = check_data(node)

    if result.in_spec:
        return False

    if result.bad_data:
        recalibrated = [diagnose(n) for n in node.dependencies]
        if not any(recalibrated):
            return False

    result = calibrate(node)
    update_parameters(result)
    return True


def maintain(node: Cal) -> None:
    for n in node.dependencies:
        maintain(n)

    result = check_state(node)
    if result.success:
        return

    result = check_data(node)
    if result.in_spec:
        return
    elif result.bad_data:
        for n in node.dependencies:
            diagnose(n)

    result = calibrate(node)
    update_parameters(result)
    return
