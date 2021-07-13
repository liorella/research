from collections import namedtuple
from typing import List, Tuple
import numpy as np
import networkx as nx

FreqNode = namedtuple("FreqNode", "frequency type")

InputNode = namedtuple("InputNode", "")

OutputNode = namedtuple("OutputNode", "")

FreqPortEdge = namedtuple("FreqPortEdge", "freq port num_channels")


class ImplDescription:
    def __init__(self,
                 freq_nodes: List[FreqNode],
                 input_nodes: List[InputNode],
                 output_nodes: List[OutputNode],
                 in_freq_edges: List[Tuple],  # todo: turn this into named tuple for self-documentation
                 out_freq_edges: List[Tuple],
                 feedback_mat: List[Tuple]):
        self._input_nodes = input_nodes
        self._freq_nodes = freq_nodes
        self._output_nodes = output_nodes

        self._create_input_graph(in_freq_edges)

        self._create_output_graph(out_freq_edges)
        
        # add attributes
        self._add_attributes(self._input_graph, 'p', input_nodes)
        self._add_attributes(self._input_graph, 'f', freq_nodes)
        self._add_attributes(self._output_graph, 'f', )
        
        

    @staticmethod
    def _add_attributes(graph, prefix, nodes):
        for i, node in enumerate(nodes):
            for attr in node._fields:
                graph.nodes[prefix + str(i)][attr] = node._asdict()[attr]

    def _create_input_graph(self, in_freq_edges):
        edgelist = [(f'p{edge[1]}', f'f{edge[0]}', {'weight': edge[2]}) for edge in in_freq_edges]
        self._input_graph = nx.from_edgelist(edgelist, create_using=nx.DiGraph)

    def _create_output_graph(self, out_freq_edges):
        edgelist = [(f'f{edge[0]}', f'p{edge[1]}', {'weight': edge[2]}) for edge in out_freq_edges]
        self._output_graph = nx.from_edgelist(edgelist, create_using=nx.DiGraph)

    @property
    def resources(self):
        # todo
        return None

    def plot_input_graph(self):
        nx.nx_pydot.write_dot(self._input_graph, 'test-output/test1.gv')

    @property
    def input_matrix(self):
        port_nodes = [node for node in self._input_graph if node[0] == 'p']
        freq_nodes = [node for node in self._input_graph if node[0] == 'f']
        nodelist = port_nodes + freq_nodes
        return nx.to_pandas_adjacency(self._input_graph, nodelist=nodelist).iloc[:len(port_nodes), len(freq_nodes):]

    @property
    def output_matrix(self):
        port_nodes = [node for node in self._output_graph if node[0] == 'p']
        freq_nodes = [node for node in self._output_graph if node[0] == 'f']
        nodelist = port_nodes + freq_nodes
        return nx.to_pandas_adjacency(self._input_graph, nodelist=nodelist)#.iloc[:len(port_nodes), len(freq_nodes):]

    def _create_in_freq_mat(self, in_freq_edges):
        in_freq_mat = np.zeros((self._num_freqs, self._num_inputs))
        for edge in in_freq_edges:
            in_freq_mat[edge[0], edge[1]] = edge[2]
        return in_freq_mat

    def _create_out_freq_mat(self, out_freq_edges):
        out_freq_mat = np.zeros((self._num_freqs, self._num_inputs))
        for edge in out_freq_edges:
            out_freq_mat[edge[0], edge[1]] = edge[2]
        return out_freq_mat


if __name__ == "__main__":
    in_edge_list = [FreqPortEdge(0, 0, 2),
                    FreqPortEdge(1, 0, 2),
                    FreqPortEdge(1, 1, 2),
                    FreqPortEdge(2, 2, 2)]
    idesc = ImplDescription([FreqNode(3.4, "q"), FreqNode(4.1, "q"), FreqNode(1.3, "r")],
                            [InputNode() for _ in range(3)],
                            [OutputNode()],
                            in_edge_list,
                            [FreqPortEdge(2, 0, 2)],
                            None)

    print(idesc.input_matrix)
    # print(idesc.output_matrix)
    # idesc.plot_input_graph()
