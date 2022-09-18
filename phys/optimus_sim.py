# simulate optimus traversal, based on https://arxiv.org/pdf/1803.03226.pdf

# graph
from enum import Enum
from logging import log


class State(Enum):
    out_of_spec = 0
    in_spec = 1
    bad_data = 2


class Cal:
    def __init__(self, name, dependencies):
        self.name = name
        self.state = State.out_of_spec
        self.last_checked_ago = None
        self.staleness = None
        self.dependencies = dependencies
        self.cal_fails = None

    def __repr__(self):
        return f"<name: {self.name} | last checked: {self.last_checked_ago} | staleness: {self.staleness}>"


class Result:
    def __init__(self):
        self.success = None
        self.in_spec = None
        self.bad_data = None


def check_state(node: Cal) -> Result:
    res = Result()
    stale = node.last_checked_ago > node.staleness
    recalibrated_dependencies = any(d.last_checked_ago < node.last_checked_ago
                                    for d in node.dependencies)
    cal_failures = node.cal_fails
    failing_dependencies = any(d.state == State.out_of_spec for d in node.dependencies)
    res.success = not (stale or recalibrated_dependencies or cal_failures or failing_dependencies)
    return res


def check_data(node: Cal) -> Result:
    res = Result()
    if node.state == State.in_spec:
        res.in_spec = True
        res.bad_data = False
    elif node.state == State.out_of_spec:
        res.in_spec = False
        res.bad_data = False
    elif node.state == State.bad_data:
        res.in_spec = False
        res.bad_data = True
    return res


def calibrate(node: Cal) -> Result:
    res = Result()
    if node.cal_fails:
        res.in_spec = State.bad_data  # NOT SURE
    else:
        res.in_spec = State.in_spec
    return Result()


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
