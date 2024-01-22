#%%

true_heads = ["0.1", "3.0", "0.10", "2.2", "4.11", "5.5", "6.9", "5.8", "5.9", "7.3", "7.9", "8.6", "8.10", "10.7", "11.10", "9.9", "9.6", "10.0", "9.0", "9.7", "10.1", "10.2", "10.6", "10.10", "11.2", "11.9"]
true_heads = set(true_heads)

def check_list(l):
    l = set(l)
    tp = true_heads.intersection(l)
    ntp = len(tp)
    fp = l.difference(true_heads)
    nfp = len(fp)
    if len(l) != 0 and len(true_heads) != 0:
        print(f"proportions: ntp: {ntp/len(true_heads)}, nfp: {nfp/len(l)}")
    print(f"ntp: {ntp}, nfp: {nfp}, total: {len(l)}, true total: {len(true_heads)}")
    return tp, fp

# l = ["0.1","0.10","3.0","5.5","7.9","8.6","8.10","9.6","9.8","9.9","10.0","10.1","10.2","10.3","10.10","11.2"]

# l = ["0.1","0.4","0.5","0.10","1.11","3.0","5.5","5.10","7.9","8.6","8.10","9.6","9.7","9.9","10.2","10.3","10.6","10.7","10.10","11.2","11.3","11.6","11.9","11.10"]
# print(check_list(l))






CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}


class NodeGroup():
    def __init__(self, name, nodes = set(), optional=[]):
        self.name = name
        self.nodes = nodes
        self.connections = set()

    def edges(self):
        s = set()
        for node in self.nodes:
            for other_group in self.connections:
                for other_node in other_group.nodes:
                    s.add((node, other_node))
        return s
    
    def recursive_edges(self):
        s = set()
        for node in self.nodes:
            for other_group in self.connections:
                for other_node in other_group.nodes:
                    s.add((node, other_node))
                    s = s | other_group.recursive_edges()
        return s


    def __lshift__(self, other):
        other.connections.add(self)
        return self

    def __rshift__(self, other):
        self.connections.add(other)
        return self
                



# "name mover":       
# "s2 inhibition":    
# "induction":        
# "duplicate token":  
# "previous token":   
# "negative":         

from typing import Any
import transformer_lens as tl





NNMH = NodeGroup(
    "negative", [(10, 7), (11, 10)]
)
BNMH = NodeGroup(
    "backup name mover",
    [
        (9, 0), 
        (9, 7), 
        (10, 1), 
        (10, 2), 
        (10, 6), 
        (10, 10), 
        (11, 2), 
        (11, 9)
    ]
)
NMH = NodeGroup(
    "name mover",
    [(9, 6), (9, 9), (10, 0)],
)
SI = NodeGroup(
    "s2 inhibition",
    [(7, 3), (7, 9), (8, 6), (8, 10)],
)
IND = NodeGroup(
    "induction",
    [(5, 5), (5, 9)],
)
DUP = NodeGroup(
    "duplicate token",
    [(3, 0), (0, 10)],
)
PREV = NodeGroup(
    "previous token",
    [(2, 2), (4, 11)],
)
RES0 = NodeGroup(
    "residual stream",
    ["resid"],
)
OUT = NodeGroup(
    "output",
    ["out"]
)

ioi_circuit = (NNMH, BNMH, NMH, SI, IND, DUP, PREV, RES0, OUT)

def edges(circuit):
    s = set()
    for node in circuit:
        s = s | node.edges()
    return s

# DEFINITION OF IOI CIRCUIT
RES0 >> PREV \
    >> SI \
    >> DUP \
    >> NMH \
    >> NNMH \
    >> BNMH \

IND << DUP \
    << PREV

SI << IND

SI >> NMH \
    >> NNMH \
    >> BNMH \

OUT << NMH \
    << NNMH \
    << BNMH \


# CIRCUIT = {
#     "name mover": [
#         (9, 9),  # by importance
#         (10, 0),
#         (9, 6),
#         (10, 10),
#         (10, 6),
#         (10, 2),
#         (10, 1),
#         (11, 2),
#         (9, 7),
#         (9, 0),
#         (11, 9),
#     ],
#     "negative": [(10, 7), (11, 10)],
#     "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
#     "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
#     "duplicate token": [
#         (0, 1),
#         (0, 10),
#         (3, 0),
#         # (7, 1),
#     ],  # unclear exactly what (7,1) does
#     "previous token": [
#         (2, 2),
#         # (2, 9),
#         (4, 11),
#         # (4, 3),
#         # (4, 7),
#         # (5, 6),
#         # (3, 3),
#         # (3, 7),
#         # (3, 6),
#     ],
# }


#%%

import parameters
from load_run import model

#%%
ioi_edges = edges(ioi_circuit)
#%%

class Circuit():
    def __init__(self, groups = []):
        self.groups = {}
        self.node_to_group = {} # TODO
        self.edges = set()
        self.srcs = {}
        self.dests = {}
        for group in groups:
            if isinstance(group, str):
                if group in self.__dict__ or group in Circuit.__dict__:
                    raise ValueError(f"Group name {group} shadows a method or attribute of Circuit")

    def __getattribute__(self, __name: str):
        if __name in object.__getattribute__(self, "groups"):
            return object.__getattribute__(self, "groups")[__name]
        else:
            return object.__getattribute__(self, __name)

    def add(self, edge):
        if isinstance(edge, set):
            for e in edge:
                self.add(e)
            return
        src, dest = edge
        if src not in self.srcs:
            self.srcs[src] = set()
        if dest not in self.dests:
            self.dests[dest] = set()
        self.srcs[src].add(dest)
        self.dests[dest].add(src)
        self.edges.add(edge)

    def update_from_groups(self):
        for group in self.groups.values():
            edges = group.edges()
            for edge in edges:
                self.add(edge)

    def inspect_node(self, node):
        print(f"Node: {node}")
        print(f"Sources: {self.dests[node]}")
        print(f"Destinations: {self.srcs[node]}")
    



def fp_tp_edges(patcher, circuit = ioi_circuit, threshold = 0.5):
    p_edges = patcher.get_edges(threshold=threshold)
    ioi_edges = edges(circuit)
    p_edges = set(p_edges)
    if len(p_edges) < 150:
        print(f"Difference: {p_edges.difference(ioi_edges)}")
        print(f"Intersection: {p_edges.intersection(ioi_edges)}")
    print(f"p_edges: {len(p_edges)}")
    print(f"ioi_edges: {len(ioi_edges)}")
    print(f"intersection: {len(p_edges.intersection(ioi_edges))}")
    if len(p_edges) != 0:
        print(f"EDGE proportions: tp: {len(p_edges.intersection(ioi_edges))/len(ioi_edges)}, fp: {len(p_edges.difference(ioi_edges))/len(p_edges)}")


if __name__ == "__main__":
    patcher :parameters.InterpolatedPathPatch = parameters.InterpolatedPathPatch.load_by_version(
        parameters.InterpolatedPathPatch.get_latest_version(), model
    )
    tp, fp = check_list(patcher.get_heads(threshold=0))
    fp_tp_edges(patcher)
    print(f"fp: {fp}")
    heads = patcher.get_heads()
    
    found_circuit = Circuit()
    p_edges = patcher.get_edges(threshold=0)
    found_circuit.add(set(p_edges))
    found_circuit.inspect_node((9, 9))
    found_circuit.inspect_node((1, 7))


# %%
