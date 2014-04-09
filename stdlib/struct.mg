def pack(fmt: str, *args):
    # Only little-endian for now
    assert fmt[0] == '<'
    bytes = []
    for [spec, member] in zip(slice(fmt, 1, len(fmt)), args):
        spec_table = {
            'b': 1,
            'h': 2,
            'i': 4,
            'q': 8
        }
        if spec not in spec_table:
            error('unsupported format specifier character: ' + spec)
        size = spec_table[spec]
        max = 1 << (size * 8 - 1)
        if member < -max or member >= max:
            error('value out of range for specifier ' + spec + ': ' + member)

        for i in range(size):
            bytes = bytes + [member >> (i * 8) & 255]

    return bytes
