def pack(fmt: str, *args):
    # Only little-endian for now
    assert fmt[0] == '<'
    bytes = []
    for [spec, member] in zip(slice(fmt, 1, None), args):
        spec_table = {
            'b': 1,
            'h': 2,
            'i': 4,
            'q': 8
        }
        if spec.isupper():
            spec = spec.lower()
            signed = False
        else:
            signed = True

        if spec not in spec_table:
            error('unsupported format specifier character: ' + spec)
        size = spec_table[spec]

        if signed:
            max = (1 << (size * 8 - 1)) - 1
            min = -max - 1
        else:
            [min, max] = [0, (1 << size * 8) - 1]
        if member < min or member > max:
            error('value out of range for specifier ' + spec + ': ' + str(member))

        # For now, since Python has infinite-bit integers, we don't need any
        # special handling for signed/unsigned
        for i in range(size):
            bytes = bytes + [member >> (i * 8) & 255]

    return bytes
