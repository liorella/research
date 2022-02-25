import itertools

from qm import ControllerConnection, InterOpxChannel


def create_fully_connected(kind=None, controllers=None):
    if kind is None:
        if controllers is None:
            controllers = ['con1', 'con2', 'con3', 'con4']
        # elif len(controllers) != 4:
        #     raise NotImplementedError('currently only 4 controllers is supported')

        connections_list = []
        max_used_port = {con: 0 for con in controllers}
        for con1, con2 in itertools.combinations(controllers, 2):
            for _ in range(2):
                connections_list.append(ControllerConnection(InterOpxChannel(con1, max_used_port[con1]),
                                                             InterOpxChannel(con2, max_used_port[con2])))
                # print(f'adding controller connection with inter opx channel {con1} port {max_used_port[con1]}'
                #       f' and {con2} port {max_used_port[con2]}')
                max_used_port[con1] += 1
                max_used_port[con2] += 1
    elif kind == 'sycamore':
        connections_list = [
            ControllerConnection(InterOpxChannel('con1', 0), InterOpxChannel('con2', 0)),
            ControllerConnection(InterOpxChannel('con1', 1), InterOpxChannel('con2', 1)),
            ControllerConnection(InterOpxChannel('con1', 2), InterOpxChannel('con2', 2)),
            ControllerConnection(InterOpxChannel('con1', 3), InterOpxChannel('con2', 3)),
            ControllerConnection(InterOpxChannel('con1', 4), InterOpxChannel('con4', 0)),
            ControllerConnection(InterOpxChannel('con1', 5), InterOpxChannel('con4', 1)),
            ControllerConnection(InterOpxChannel('con1', 6), InterOpxChannel('con3', 0)),
            ControllerConnection(InterOpxChannel('con1', 7), InterOpxChannel('con3', 1)),
            ControllerConnection(InterOpxChannel('con2', 4), InterOpxChannel('con4', 2)),
            ControllerConnection(InterOpxChannel('con2', 5), InterOpxChannel('con4', 3)),
            ControllerConnection(InterOpxChannel('con2', 6), InterOpxChannel('con3', 2)),
            ControllerConnection(InterOpxChannel('con2', 7), InterOpxChannel('con3', 3)),
            ControllerConnection(InterOpxChannel('con3', 4), InterOpxChannel('con4', 4)),
            ControllerConnection(InterOpxChannel('con3', 5), InterOpxChannel('con4', 5)),
            ControllerConnection(InterOpxChannel('con3', 6), InterOpxChannel('con4', 6)),
            ControllerConnection(InterOpxChannel('con3', 7), InterOpxChannel('con4', 7)),
        ]
    else:
        raise ValueError(f"unknown kind {kind}")
    return connections_list

