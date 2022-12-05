from typing import Optional


def dict_or_namedtuple_get(data, key):
    r"""Get item from dict or namedtuple."""
    try:
        v = getattr(data, key)
    except AttributeError:
        v = data[key]
    return v


def get_dict_or_namedtuple_keys(container):
    r"""Get keys/fields of a dict/namedtuple."""
    try:
        keys = container._fields
    except AttributeError:
        keys = container.keys()
    return list(keys)


def is_node(node_or_edge):
    r"""Determine whether the input string represents a node."""
    return node_or_edge in {'nuclei', 'electrons'}


def is_edge(node_or_edge):
    r"""Determine whether the input string represents an edge."""
    return node_or_edge in {'nn', 'ne', 'en', 'same', 'anti'}


class NodeEdgeMapping:
    r"""A utility mapping between the various node and edge types.

    For example it is often useful determine the sender/receiver node type of a given
    edge, or generate all the edge types with a given sender/receiver node.

    Args:
        edges (Sequence[str]): all the edge types present in the graph
        node_data (dict): optional, data to store for the node types
    """

    def __init__(self, edges, node_data: Optional[dict] = None):
        self.edges = edges
        self.nodes = {self.receiver_of(edge) for edge in edges}
        self.node_data = node_data

    def get_data_container(self, data):
        return self.node_data[data] if isinstance(data, str) else data

    def with_receiver(self, node_or_edge):
        if is_edge(node := node_or_edge):
            return [node_or_edge]
        return [edge for edge in self.edges if self.receiver_of(edge) == node]

    def with_sender(self, node_or_edge):
        if is_edge(node := node_or_edge):
            return [node_or_edge]
        return [edge for edge in self.edges if self.sender_of(edge) == node]

    def data_with_receiver(self, node_or_edge, data):
        edges = self.with_receiver(node_or_edge)
        return [self.edge_data_of(edge, data) for edge in edges]

    def data_with_sender(self, node_or_edge, data):
        edges = self.with_sender(node_or_edge)
        return [self.edge_data_of(edge, data) for edge in edges]

    def node_or_receiver_data_of(self, node_or_edge, data):
        data_of = self.node_data_of if is_node(node_or_edge) else self.receiver_data_of
        return data_of(node_or_edge, data)

    def node_or_sender_data_of(self, node_or_edge, data):
        data_of = self.node_data_of if is_node(node_or_edge) else self.sender_data_of
        return data_of(node_or_edge, data)

    def reduce_to_receiver(self, node, data, reduce_fn):
        data_container = self.get_data_container(data)
        keys = get_dict_or_namedtuple_keys(data_container)
        if node in keys:
            return dict_or_namedtuple_get(data_container, node)
        return reduce_fn(self.data_with_receiver(node, data_container))

    def receiver_data_of(self, edge, data):
        node = self.receiver_of(edge)
        return self.node_data_of(node, data)

    def sender_data_of(self, edge, data):
        node = self.sender_of(edge)
        return self.node_data_of(node, data)

    def node_data_of(self, node, data):
        data_container = self.get_data_container(data)
        return dict_or_namedtuple_get(data_container, node)

    def edge_data_of(self, edge, data):
        return dict_or_namedtuple_get(data, edge)

    def receiver_of(self, edge):
        node = {
            'same': 'electrons',
            'anti': 'electrons',
            'ne': 'electrons',
            'en': 'nuclei',
            'nn': 'nuclei',
        }[edge]
        return node

    def sender_of(self, edge):
        node = {
            'same': 'electrons',
            'anti': 'electrons',
            'ne': 'nuclei',
            'en': 'electrons',
            'nn': 'nuclei',
        }[edge]
        return node
