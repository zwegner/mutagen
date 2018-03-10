EDGE_REG = 0
EDGE_LIST = 1
EDGE_DICT = 2

class DirectedGraph(nodes: list = [], edge_values: dict = {}, uses: dict = {}, edge_types: dict = {}, node_seq_id=0):
    def get(self, node, edge):
        return self.edge_values[node][edge]

    def add_node(self, node):
        if node in self.nodes:
            return self
        return (self <- .edge_values[node] = {},
            .uses[node] = [],
            .nodes += [node])

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

    def delete_node(self, node):
        # First, delete all uses of other nodes that this node has
        uses = {other_node: [u for u in usages if u[1] != node]
                for [other_node, usages] in self.uses}

        return (self <- .nodes = [n for n in self.nodes if n != node],
                .edge_values = self.edge_values - [node],
                .uses = uses - [node])

    def add_edge(self, node, edge, type):
        if [node, edge] in self.edge_types:
            assert self.edge_types[[node, edge]] == type
            return self
        if type == EDGE_LIST:
            self = self <- .edge_values[node][edge] = []
        elif type == EDGE_DICT:
            self = self <- .edge_values[node][edge] = {}
        return (self <- .edge_types[[node, edge]] = type)

    def get_uses(self, node):
        return self.uses[node]

    def remove_use(self, node, usage):
        # XXX Kinda gross, we can add duplicate usages in some cases to avoid deleting nodes in
        # right before we add a reference to them, so only delete a single use here
        found_index = None
        for [index, use] in enumerate(self.uses[node]):
            if use == usage:
                found_index = index
                break
        assert found_index != None

        self = self <- .uses[node] = self.uses[node][:found_index] + self.uses[node][found_index + 1:]

        # Unreferenced node: delete it
        if not self.uses[node]:
            self = self.delete_node(node)
        return self

    def set_edge(self, node, edge, value):
        self = self.add_edge(node, edge, EDGE_REG)
        usage = ['reg', node, edge]

        # Add the new usage before we remove any to avoid premature deletion
        self = self <- .uses[value] += [usage]

        # Remove the usage of any old value across this edge
        if edge in self.edge_values[node]:
            self = self.remove_use(self.edge_values[node][edge], usage)

        return self <- .edge_values[node][edge] = value

    def create_edge_list(self, node, edge):
        return self.add_edge(node, edge, EDGE_LIST)

    def set_edge_list(self, node, edge, value):
        self = self.add_edge(node, edge, EDGE_LIST)

        # Add the new usage before we remove any to avoid premature deletion
        for [index, item] in enumerate(value):
            self = self <- .uses[item] += [['list', node, edge, index]]

        # Remove the usage of every value in this list, if any
        for [index, item] in enumerate(self.edge_values[node][edge]):
            self = self.remove_use(item, ['list', node, edge, index])

        return (self <- .edge_values[node][edge] = value)

    def set_edge_index(self, node, edge, index, value, _append=False):
        self = self.add_edge(node, edge, EDGE_LIST)
        # Handle appends by just appending a None for now that will get overwritten immediately
        if _append:
            edge_list = self.edge_values[node][edge]
            index = len(edge_list)
            self = self <- .edge_values[node][edge] = edge_list + [None]
        # Otherwise, remove the usage of the old value at this index
        else:
            self = self.remove_use(self.edge_values[node][edge][index], ['list', node, edge, index])

        return (self <- .edge_values[node][edge][index] = value,
            .uses[value] += [['list', node, edge, index]])

    def append_to_edge(self, node, edge, value):
        return self.set_edge_index(node, edge, None, value, _append=True)

    def create_edge_dict(self, node, edge):
        return self.add_edge(node, edge, EDGE_DICT)

    def unset_edge_key(self, node, edge, key):
        self = self.add_edge(node, edge, EDGE_DICT)

        # Remove the usage of the old value at this key
        if key in self.edge_values[node][edge]:
            self = self.remove_use(self.edge_values[node][edge][key], ['value', node, edge, key])

        return self <- .edge_values[node][edge] -= [key]

    def set_edge_key(self, node, edge, key, value):
        self = self.add_edge(node, edge, EDGE_DICT)
        usage = ['value', node, edge, key]

        # Add the new usage before we remove any to avoid premature deletion
        self = self <- .uses[value] += [usage]

        # Remove the usage of the old value at this key
        if key in self.edge_values[node][edge]:
            self = self.remove_use(self.edge_values[node][edge][key], usage)

        return self <- .edge_values[node][edge][key] = value
            # XXX
            #.uses[key] = self.uses[key] + [['key', node, edge]],
