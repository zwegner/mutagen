EDGE_REG = 0
EDGE_LIST = 1
EDGE_DICT = 2

class DirectedGraph(nodes: list = [], edge_sets: dict = {}, uses: dict = {}, edge_types: dict = {}, node_seq_id=0):
    def get(self, node, edge):
        return self.edge_sets[node][edge]

    def add_node(self, node):
        if node in self.nodes:
            return self
        return (self <- .edge_sets[node] = {},
            .uses[node] = [],
            .nodes = self.nodes + [node])

    def replace_node(self, old_node, new_node, exclude_uses_from=None):
        for usage in self.uses[old_node]:
            [type, usage] = [usage[0], usage[1:]]
            if exclude_uses_from and usage[0] in exclude_uses_from:
                continue
            if type == 'reg':
                self = self.set_edge(*usage, new_node)
            elif type == 'list':
                self = self.set_edge_index(*usage, new_node)
            elif type == 'value':
                self = self.set_edge_key(*usage, new_node)
            else:
                assert False
        return self

    def add_edge(self, node, edge, type):
        if [node, edge] in self.edge_types:
            assert self.edge_types[[node, edge]] == type
            return self
        if type == EDGE_LIST:
            self = self <- .edge_sets[node][edge] = []
        elif type == EDGE_DICT:
            self = self <- .edge_sets[node][edge] = {}
        return (self <- .edge_types[[node, edge]] = type)

    def get_uses(self, node):
        return self.uses[node]

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
            self = self.remove_use(self.edge_sets[node][edge], ['reg', node, edge])

        return (self <- .edge_sets[node][edge] = value,
            .uses[value] = self.uses[value] + [['reg', node, edge]])

    def create_edge_list(self, node, edge):
        return self.add_edge(node, edge, EDGE_LIST)

    def set_edge_list(self, node, edge, value):
        self = self.add_edge(node, edge, EDGE_LIST)

        # Remove the usage of every value in this list, if any
        for item in self.edge_sets[node][edge]:
            self = self.remove_use(item, ['reg', node, edge])

        for [index, item] in enumerate(value):
            self = self <- .uses[item] = self.uses[item] + [['list', node, edge, index]]

        return (self <- .edge_sets[node][edge] = value)

    def set_edge_index(self, node, edge, index, value, _append=False):
        self = self.add_edge(node, edge, EDGE_LIST)
        # Handle appends by just appending a None for now that will get overwritten immediately
        if _append:
            edge_list = self.edge_sets[node][edge]
            index = len(edge_list)
            self = self <- .edge_sets[node][edge] = edge_list + [None]
        # Otherwise, remove the usage of the old value at this index
        else:
            self = self.remove_use(self.edge_sets[node][edge][index], ['list', node, edge, index])

        return (self <- .edge_sets[node][edge][index] = value,
            .uses[value] = self.uses[value] + [['list', node, edge, index]])

    def append_to_edge(self, node, edge, value):
        return self.set_edge_index(node, edge, None, value, _append=True)

    def create_edge_dict(self, node, edge):
        return self.add_edge(node, edge, EDGE_DICT)

    def set_edge_key(self, node, edge, key, value):
        self = self.add_edge(node, edge, EDGE_DICT)
        # Remove the usage of the old value at this key
        if key in self.edge_sets[node][edge]:
            self = self.remove_use(self.edge_sets[node][edge][key], ['value', node, edge, key])

        return (self <- .edge_sets[node][edge][key] = value,
            # XXX
            #.uses[key] = self.uses[key] + [['key', node, edge]],
            .uses[value] = self.uses[value] + [['value', node, edge, key]])
