# Doesn't get much more basic than this

@test_fn(1)
def basic():
    return 1

# Test static data

data_0_7 = static_string('01234567')

@test_fn(0x30)
def test_static_data_8():
    return movzx(address(data_0_7, 8))

@test_fn(0x3736353433323130)
def test_static_data_64():
    return mov(address(data_0_7, 64))

# Function calls 1: big nested mess with multiple blocks, inlining, etc.

@test_fn(16)
def fn_calls_1():
    # Also export the nested function so we can test the non-inlined version
    @export
    def t1(a):
        b = 0
        if a:
            b = a + 1
        def t2(c):
            return b + c + 2
        return t2(4) + 1
    return t1(0) + t1(1)

# Function calls 2: same as 1, but with t1 not inlined

@test_fn(16)
def fn_calls_2():
    t1 = _extern_label('t1')
    return t1(0) + t1(1)

# Function calls 3: calling an exported function directly/indirectly

@export
def exported(a):
    return a + 1

@test_fn(3)
def fn_calls_3():
    _exported = _extern_label('exported')
    return exported(0) + _exported(1)

# Function calls 4: test that constant values can be inlined into functions

four = 4
@export
def f_four():
    return four

@test_fn(4)
def fn_calls_4():
    return f_four()
