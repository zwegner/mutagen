# ELF64 object creator. Rather complicated for its simple purpose.
# Reference: http://downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf
import struct

def pack(spec, *args):
    return list(struct.pack(spec, *args))

def create_string_table(strings):
    table = [0]
    offsets = []
    for s in strings:
        offsets = offsets + [len(table)]
        table = table + list(s.encode('ascii')) + [0]
    return [table, offsets]

def create_elf_file(code, local_labels, global_labels, extern_labels):
    section_names = ['.text', '.shstrtab', '.strtab', '.symtab', '.reltext']
    section_types = [1, 3, 3, 2, 9]
    [code_idx, shstrtab_idx, strtab_idx, symtab_idx] = list(range(1, 5))

    [shstrtab, shstrtab_offsets] = create_string_table(section_names)

    strings = [label for [label, address] in local_labels + global_labels + extern_labels]
    [strtab, strtab_offsets] = create_string_table(strings)

    # Create symbol table section--just a list of labels for now
    symtab = [0] * 24 # First symbol is reserved
    i = 0
    for [flag, labels] in [[0, local_labels], [0x10, global_labels], [0x20, extern_labels]]:
        for [label, address] in labels:
            if flag == 0x20:
                section = address = 0
            else:
                section = code_idx
            symtab = symtab + pack('<IBBHQQ',
                strtab_offsets[i], # Name offset in string table
                flag, # type/binding (for us, specify local or global)
                0, # reserved/unused
                section, # section index of definition
                address, # value of symbol (offset into code section)
                0 # size
            )
            i = i + 1

    # Create relocation table
    relocations = []
    extern_sym_idx = len(local_labels) + len(global_labels) + 1
    for [label, address] in extern_labels:
        relocations = relocations + pack('<QII', address, 2, extern_sym_idx)
        extern_sym_idx = extern_sym_idx + 1

    sections = [code, shstrtab, strtab, symtab, relocations]

    elf_header = list('\x7fELF'.encode('ascii')) + [ # magic
        2, # class (elf64)
        1, # data format (little endian)
        1, # elf version
        0, # OS ABI (sysV)
        0, # ABI version
        0, 0, 0, 0, 0, 0, 0 # padding
    ] + pack('<HHIQQQIHHHHHH',
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
        shstrtab_idx # section index of section name string table
    )

    reserved_section = [0] * 64

    elf_headers = elf_header + reserved_section
    data_offset = len(elf_headers) + 64 * len(sections)
    elf_data = []
    for [i, [data, section_type]] in enumerate(zip(sections, section_types)):
        if section_type == 2: # .symtab has special handling
            [link, alignment, size, info] = [strtab_idx, 4, 24, len(local_labels)]
        elif section_type == 9:
            [link, alignment, size, info] = [symtab_idx, 1, 16, code_idx]
        else:
            [link, alignment, size, info] = [0, 1, 0, 0]
        section_header = pack('<IIQQQQIIQQ',
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
        elf_headers = elf_headers + section_header
        elf_data = elf_data + data

    return elf_headers + elf_data
