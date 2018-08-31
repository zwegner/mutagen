class GetRandomInt64Effect:
    pass

def rol(x, y):
    return (x << y) | (x >> 64 - y)

def int64(x):
    return x & (1 << 64) - 1

def get_default_random_seed():
    return [0x8C84A911159F4017, 0x062C0B602809C02E, 0xA48B831518DEA5D7,
            0x55AB3636D17F3AD3]

# RKISS algorithm
def gen_rand_64(state):
    [a, b, c, d] = state
    e = a - rol(b, 7)
    a = b ^ rol(c, 13)
    b = c + rol(d, 37)
    c = d + e
    d = e + a
    result = int64(d)
    state = [int64(a), int64(b), int64(c), result]
    return [result, state]

def handle_randomness(fn):
    state = get_default_random_seed()
    consume:
        return fn()
    effect GetRandomInt64Effect as e:
        [r, state] = gen_rand_64(state)
        resume r
