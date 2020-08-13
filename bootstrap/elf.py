# ELF64 object creator. Rather complicated for its simple purpose.
# Reference: http://downloads.openwatcom.org/ftp/devel/docs/elf-64-gen.pdf
import struct

def pack(spec, *args):
    return list(struct.pack(spec, *args))

def create_string_table(strings):
    table = [0]
    offsets = {}
    for s in strings:
        offsets[s] = len(table)
        table = table + list(s.encode('ascii')) + [0]
    return [table, offsets]

def create_elf_file(code, data, global_labels, extern_labels):
    section_names = ['.text', '.rodata', '.shstrtab', '.strtab', '.symtab',
            '.reltext', '.relrodata']
    section_types = [1, 1, 3, 3, 2, 9, 9]
    [code_idx, data_idx, shstrtab_idx, strtab_idx, symtab_idx, reltext_idx,
            relrodata_idx] = list(range(1, len(section_names)+1))

    section_idx = {'code': code_idx, 'data': data_idx}

    [shstrtab, shstrtab_offsets] = create_string_table(section_names)

    label_set = {label for labels in [global_labels, extern_labels]
            for [label, _, _] in labels}

    strings = list(sorted(label_set))
    [strtab, strtab_offsets] = create_string_table(strings)

    # Create symbol table section--just a list of labels for now
    symtab = [0] * 24 # First symbol is reserved
    for [flag, labels] in [[0x10, global_labels], [0x20, extern_labels]]:
        for [label, section, address] in labels:
            if flag == 0x20:
                section = address = 0
            else:
                section = section_idx[section]
            symtab = symtab + pack('<IBBHQQ',
                strtab_offsets[label], # Name offset in string table
                flag, # type/binding (for us, specify local or global)
                0, # reserved/unused
                section, # section index of definition
                address, # value of symbol (offset into code section)
                0 # size
            )

    # Create relocation table
    relocations = {}
    extern_sym_idx = len(global_labels) + 1
    for rel_type in ['code', 'data']:
        relocations[rel_type] = []
        for [label, section, address] in extern_labels:
            if section == rel_type:
                relocations[rel_type] += pack('<QII', address, 2, extern_sym_idx)
                extern_sym_idx = extern_sym_idx + 1

    sections = [code, data, shstrtab, strtab, symtab, relocations['code'], relocations['data']]

    elf_header = list(b'\x7fELF') + [ # magic
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
    for [name, data, section_type] in zip(section_names, sections, section_types):
        if section_type == 2: # .symtab has special handling
            [link, alignment, size, info] = [strtab_idx, 4, 24, 0]
        elif section_type == 9:
            # Ugh
            rel_idx = code_idx if name == '.reltext' else data_idx
            [link, alignment, size, info] = [symtab_idx, 1, 16, rel_idx]
        else:
            # Use 64-byte alignment for code/data, which is about the most we'd
            # care about generally...
            [link, alignment, size, info] = [0, 64, 0, 0]
        section_header = pack('<IIQQQQIIQQ',
            shstrtab_offsets[name], # section name in section name string table
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
