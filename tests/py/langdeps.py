# Compatibility layer for differences in Python/Mutagen

def assert_call_fails(fn, *args):
    try:
        fn(*args)
    except:
        return True
    assert False
