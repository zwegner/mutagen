EDGE_REG = 0
EDGE_LIST = 1

class DirectedGraph(nodes: list = [], edge_sets: dict = {}, uses: dict = {}, edge_types: dict = {}):
    def get(self, node, edge):
        return self.edge_sets[node][edge]

    def add_node(self, node):
        if node in self.nodes:
            return self
        return (self <- .edge_sets[node] = {},
            .uses[node] = [],
            .nodes = self.nodes + [node])

    def replace_node(self, old_node, new_node):
        for usage in self.uses[old_node]:
            if len(usage) == 3:
                self = self.set_edge_index(*usage, new_node)
            else:
                self = self.set_edge(*usage, new_node)
        return self

    def add_edge(self, node, edge, type):
        if [node, edge] in self.edge_types:
            assert self.edge_types[[node, edge]] == type
            return self
        if type == EDGE_LIST:
            self = self <- .edge_sets[node][edge] = []
        return (self <- .edge_types[[node, edge]] = type)

    def remove_use(self, node, usage):
        self = self <- .uses[node] = list(filter(lambda (u): u != usage, self.uses[node]))
        # Unreferenced node: delete it
        if not self.uses[node]:
            self = (self <- .nodes = list(filter(lambda (n): n != node, self.nodes)),
                    .edge_sets = self.edge_sets - [node],
                    .uses = self.uses - [node])
        return self

    def set_edge(self, node, edge, value):
        self = self.add_edge(node, edge, EDGE_REG)
        # Remove the usage of any old value across this edge
        if edge in self.edge_sets[node]:
            self = self.remove_use(self.edge_sets[node][edge], [node, edge])

        return (self <- .edge_sets[node][edge] = value,
            .uses[value] = self.uses[value] + [[node, edge]])

    def set_edge_index(self, node, edge, index, value, _append=False):
        self = self.add_edge(node, edge, EDGE_LIST)
        # Handle appends by just appending a None for now that will get overwritten immediately
        if _append:
            edge_list = self.edge_sets[node][edge]
            index = len(edge_list)
            self = self <- .edge_sets[node][edge] = edge_list + [None]
        # Otherwise, remove the usage of old value at this index
        else:
            self = self.remove_use(self.edge_sets[node][edge][index], [node, edge, index])

        return (self <- .edge_sets[node][edge][index] = value,
            .uses[value] = self.uses[value] + [[node, edge, index]])

    def append_to_edge(self, node, edge, value):
        return self.set_edge_index(node, edge, None, value, _append=True)

    # XXX this is a hacky api
    def create_node(self, new_class, *args):
        node_id = len(self.nodes)
        node = new_class(node_id, *args)
        return [self.add_node(node), node]
