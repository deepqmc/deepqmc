from collections import namedtuple

Psi = namedtuple('Psi', 'sign log')
GraphEdges = namedtuple('GraphEdges', 'senders receivers features')
GraphNodes = namedtuple('GraphNodes', 'nuclei electrons')
Graph = namedtuple('Graph', 'nodes edges')
