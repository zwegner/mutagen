# Priority queues based on Braun Heaps, based on discussion at
# http://alaska-kamtchatka.blogspot.com/2010/02/braun-trees.html
# This is actually a sane purely-functional data structure and not a complete hack!

# XXX recursion
class BraunHeap(data, left, right):
    def is_empty(self):
        return False

    def get_min(self):
        return self.data

    def replace_min(self, item):
        return type(self)(item, self.left, self.right).sift_down()

    # Make sure the root element is the lowest one, by potentially swapping it
    # with one of the children
    def sift_down(self):
        [data, left, right] = [self.data, self.left, self.right]
        if (left.is_empty() or (data < left.data and
            (right.is_empty() or data < right.data))):
            return self
        if (right.is_empty() or left.data < right.data):
            [data, left] = [left.data, left.replace_min(data)]
        else:
            [data, right] = [right.data, right.replace_min(data)]
        return type(self)(data, left, right)

    def insert(self, item):
        data = self.data
        if item >= data:
            [item, data] = [data, item]
        return type(self)(item, self.right.insert(data), self.left)

    def delete_min(self):
        if self.right.is_empty():
            return self.left
        assert self.left
        [data, right] = [self.left.data, self.right]
        if self.left.data >= self.right.data:
            [data, right] = [right.data, right.replace_min(data)]
        return type(self)(data, right, self.left.delete_min())

    # Return the difference in size between this tree and the number n, which
    # is the size of this tree's right sibling. Relies on the Braun Tree property
    # of the left/right sizes differing by at most one.
    def diff(self, n):
        if n == 0:
            return 1
        if n & 1:
            return self.left.diff((n - 1) // 2)
        return self.right.diff((n - 2) // 2)

    def size(self):
        m = self.right.size()
        return 2 * m + 1 + self.left.diff(m)

    # Helper method to make sure our heap is consistent
    def check(self):
        # Check Braun Tree invariant: sizes are equal or left is one bigger
        [ls, rs] = [self.left.size(), self.right.size()]
        assert rs <= ls and ls <= rs + 1
        for child in [self.left, self.right]:
            # Check heap invariant: minimum value is the root
            assert child.is_empty() or self.data < child.data
            # And recursively check all subtrees
            child.check()

    def __str__(self):
        return 'BraunHeap({}, {}, {})'.format(self.data, self.left, self.right)

class EmptyBraunHeap:
    def is_empty(self):
        return True
    def diff(self, n):
        assert n == 0
        return 0
    def size(self):
        return 0
    def get_min(self):
        assert False
    def replace_min(self):
        assert False
    def insert(self, item):
        return BraunHeap(item, self, self)
    def delete_min(self):
        assert False
    def check(self):
        pass
    def __str__(self):
        return '_'

Empty = EmptyBraunHeap()

# XXX recursion
class PriorityQueue(heap):
    def __init__(items):
        # HACK ugh!
        if isinstance(items, BraunHeap) or isinstance(items, EmptyBraunHeap):
            heap = items
        else:
            heap = Empty
            for item in items:
                heap = heap.insert(item)
        return {'heap': heap}
    def empty(self):
        return self.heap.is_empty()
    def put(self, item):
        return type(self)(self.heap.insert(item))
    def get(self):
        return [type(self)(self.heap.delete_min()), self.heap.get_min()]
    def __str__(self):
        return 'PriorityQueue({})'.format(self.heap)
