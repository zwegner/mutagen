# The Computer Language Benchmarks Game
# http://benchmarksgame.alioth.debian.org/
#
# contributed by Antoine Pitrou
# modified by Dominique Wahli and Daniel Nanz
# modified a bit for Mutagen syntax by Zach Wegner

def make_tree(mt, i, d):
    if d > 0:
        i2 = i + i
        return [i, mt(mt, i2 - 1, d - 1), mt(mt, i2, d - 1)]
    return [i, None, None]

def check_tree(ct, node):
    [i, l, r] = node
    if l == None:
        return i
    return i + ct(ct, l) - ct(ct, r)

def make_check(itde):
    [i, d] = itde
    return check_tree(check_tree, make_tree(make_tree, i, d))

def get_argchunks(i, d):
    chunksize = 5000
    chunk = []
    for k in range(1, i + 1):
        chunk = chunk + [[k, d], [-k, d]]
        if len(chunk) == chunksize:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def main(n):
    min_depth = 4
    max_depth = n
    stretch_depth = max_depth + 1

    print('stretch tree of depth {0}\t check: {1}'.format(
          stretch_depth, make_check([0, stretch_depth])))

    long_lived_tree = make_tree(make_tree, 0, max_depth)

    mmd = max_depth + min_depth
    for d in range(min_depth, stretch_depth, 2):
        i = 2 ** (mmd - d)
        cs = 0
        for argchunk in get_argchunks(i,d):
            for args in argchunk:
                cs = cs + make_check(args)
        print('{0}\t trees of depth {1}\t check: {2}'.format(i * 2, d, cs))

    print('long lived tree of depth {0}\t check: {1}'.format(
          max_depth, check_tree(check_tree, long_lived_tree)))

main(12)
