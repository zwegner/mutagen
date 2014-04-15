# ELF64 object creator. Rather complicated for its simple purpose.
# Reference: http://downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf
import struct

def create_string_table(strings):
    table = [0]
    offsets = []
    for s in strings:
        offsets = offsets + [len(table)]
        table = table + s.encode('ascii') + [0]
    return [table, offsets]

def create_elf_file(code, labels):
    section_names = ['.text', '.shstrtab', '.strtab', '.symtab']
    section_types = [1, 3, 3, 2]
    [shstrtab, shstrtab_offsets] = create_string_table(section_names)

    strings = []
    for [label, address] in labels:
        strings = strings + [label]
    [strtab, strtab_offsets] = create_string_table(strings)

    symtab = [0] * 24 # First symbol is reserved
    for [i, [label, address]] in enumerate(labels):
        symtab = symtab + struct.pack('<IBBHQQ',
            strtab_offsets[i], # Name offset in string table
            0, # type/binding (hardcoded to local for now)
            0, # reserved
            1, # section index of code section
            address, # value of symbol (an address)
            0 # size
        )

    sections = [code, shstrtab, strtab, symtab]

    elf_header = '\x7fELF'.encode('ascii') + [ # magic
        2, # class (elf64)
        1, # data format (little endian)
        1, # elf version
        0, # OS ABI (sysV)
        0, # ABI version
        0, 0, 0, 0, 0, 0, 0 # padding
    ] + struct.pack('<HHIQQQIHHHHHH',
        1, # file type (relocatable object file)
        62, # machine type (x86-64)
        1, # elf version
        0, # entry point
        0, # program header offset
        64, # section header offset
        0, # flags
        64, # elf header size
        0, 0, # size/number of program header entries
        64, # size of section header entry
        len(sections) + 1, # number of section header entries (+1 for reserved)
        2 # section index of section name string table
    )

    reserved_section = [0] * 64

    elf_file = elf_header + reserved_section
    data_offset = len(elf_file) + 64 * len(sections)
    elf_data = []
    for [i, [data, section_type]] in enumerate(zip(sections, section_types)):
        if section_type == 2: # .symtab has special handling
            n_local_syms = len(labels) # no globals for now
            [link, alignment, size, info] = [3, 4, 24, n_local_syms]
        else:
            [link, alignment, size, info] = [0, 1, 0, 0]
        section_header = struct.pack('<IIQQQQIIQQ',
            shstrtab_offsets[i], # section name in section name string table
            section_type, # section type
            2, # flags
            0, # starting address
            data_offset + len(elf_data), # offset in file
            len(data), # size of section
            link, # link
            info, # misc info
            alignment, # address alignment
            size # entry size
        )
        elf_file = elf_file + section_header
        elf_data = elf_data + data

    elf_file = elf_file + elf_data
    return elf_file
