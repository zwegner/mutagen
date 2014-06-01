#!/usr/bin/env python3
import os
import sys

import lexer
import liblex
import libparse

import mg_builtins
from syntax import *

# XXX
NULL_INFO = liblex.Info('???', 0)

def reduce_binop(p):
    r = p[0]
    for item in p[1]:
        r = BinaryOp(item[0], r, item[1])
    return r

def reduce_list(p):
    return [p[0]] + [item[1] for item in p[1]]

rule_table = [
    # Atoms
    ['identifier', ('IDENTIFIER', lambda p: Identifier(p[0], info=p.get_info(0)))],
    ['none', ('NONE', lambda p: None_(info=p.get_info(0)))],
    ['boolean', ('TRUE|FALSE', lambda p:
        Boolean({'True': True, 'False': False}[p[0]], info=p.get_info(0)))],
    ['integer', ('INTEGER', lambda p: Integer(p[0], info=p.get_info(0)))],
    ['string', ('STRING', lambda p: String(p[0], info=p.get_info(0)))],
    ['parenthesized', ('LPAREN test RPAREN', lambda p: p[1])],
    ['atom', 'identifier|none|boolean|integer|string|parenthesized|list_comp|'
        'dict_set_comp'],
]

# Stupid function since our parser library is stupid. Parse the tail end of
# a list/dict/set, which might have extra commas, but only at the end!
def parse_commas(items):
    for i, item in enumerate(items):
        if item[1] is not None:
            yield item[1]
        else:
            assert i == len(items) - 1

@libparse.rule_fn(rule_table, 'list_comp', 'LBRACKET [test (comp_iter+|'
    '(COMMA [test])*)] RBRACKET')
def parse_list(p):
    if p[1]:
        items = p[1]
        l = [items[0]]
        if items[1]:
            if isinstance(items[1][0], CompIter):
                return Scope(ListComprehension(items[0], items[1]))
            l += list(parse_commas(items[1]))
        return List(l, info=p.get_info(0))
    return List([], info=p.get_info(0))

# This function is a wee bit crazy but it kinda has to be that way with our
# parser and AST design
@libparse.rule_fn(rule_table, 'dict_set_comp', 'LBRACE [test (COLON test (comp_iter+|'
    '(COMMA [test COLON test])*)|'
    '(COMMA [test])*)] RBRACE')
def parse_dict_set(p):
    if p[1]:
        items = p[1]
        if items[1] and items[1][0] == ':':
            key, value = items[0], items[1][1]
            d = {key: value}
            if items[1][2]:
                items = items[1][2]
                if isinstance(items[0], CompIter):
                    return Scope(DictComprehension(key, value, items))
                d.update({item[0]: item[2] for item in parse_commas(items)})
            return Dict(d, info=p.get_info(0))
        else:
            set_items = [items[0]]
            if items[1]:
                set_items += list(parse_commas(items[1]))
            return Call(Identifier('set', info=p.get_info(0)),
                [List(set_items, info=p.get_info(0))])
    return Dict({}, info=p.get_info(0))

@libparse.rule_fn(rule_table, 'arg', 'test [EQUALS test]')
def parse_arg(p):
    if p[1]:
        assert isinstance(p[0], Identifier)
        return KeywordArg(p[0].name, p[1][1])
    return p[0]

@libparse.rule_fn(rule_table, 'subscript', '[test] [COLON [test] [COLON [test]]]]')
def parse_subscript(p):
    [start, stop, step] = [None_(info=NULL_INFO)] * 3
    if p[0]:
        if p[1] is None:
            return lambda expr: GetItem(expr, p[0])
        start = p[0]
    assert p[1]
    if p[1][1] is not None:
        stop = p[1][1]
    if p[1][2] is not None and p[1][2][1] is not None:
        step = p[1][2][1]
    return lambda expr: Call(Identifier('slice', info=NULL_INFO),
        [expr, start, stop, step])

rule_table += [
    # Function calls, subscripts, attribute accesses
    ['vararg', ('STAR test', lambda p: VarArg(p[1]))],
    ['args', ('(arg|vararg) (COMMA (arg|vararg))*', reduce_list)],
    # Since the trailer rules don't have access to the left-hand side, return lambdas
    ['call', ('LPAREN [args] RPAREN', lambda p: lambda expr: Call(expr, p[1] or []))],
    ['getitem', ('LBRACKET subscript RBRACKET', lambda p: p[1])],
    ['getattr', ('PERIOD IDENTIFIER', lambda p: lambda expr: GetAttr(expr, p[1]))],
    ['trailer', 'call|getitem|getattr'],
]

@libparse.rule_fn(rule_table, 'power', 'atom trailer* [STAR_STAR factor]')
def parse_power(p):
    r = p[0]
    for trailer in p[1]:
        r = trailer(r)
    if p[2]:
        r = BinaryOp('**', p[2][1])
    return r

rule_table += [
    # Binary ops
    ['factor', 'power', ('(PLUS|MINUS|INVERSE) factor', lambda p: UnaryOp(p[0], p[1]))],
    ['term', ('factor ((STAR|FLOORDIV|MODULO) factor)*', reduce_binop)],
    ['arith_expr', ('term ((PLUS|MINUS) term)*', reduce_binop)],
    ['shift_expr', ('arith_expr ((SHIFT_LEFT|SHIFT_RIGHT) arith_expr)*', reduce_binop)],
    ['and_expr', ('shift_expr (BIT_AND shift_expr)*', reduce_binop)],
    ['xor_expr', ('and_expr (BIT_XOR and_expr)*', reduce_binop)],
    ['or_expr', ('xor_expr (BIT_OR xor_expr)*', reduce_binop)],
    ['expr', 'or_expr'],
]

# XXX chaining comparisons?
@libparse.rule_fn(rule_table, 'comparison', 'or_expr ((EQUALS_EQUALS|NOT_EQUALS|'
    'GREATER|GREATER_EQUALS|LESS|LESS_EQUALS|IN|NOT IN) or_expr)*')
def parse_comparison(p):
    r = p[0]
    for item in p[1]:
        if item[0] == 'in':
            r = BinaryOp('in', item[1], r)
        elif item[0] == ['not', 'in']:
            r = UnaryOp('not', BinaryOp('in', item[1], r))
        else:
            r = BinaryOp(item[0], r, item[1])
    return r

rule_table += [
    ['not_test', 'comparison', ('NOT not_test', lambda p: UnaryOp('not', p[1]))],
    ['and_test', ('not_test (AND not_test)*', reduce_binop)],
    ['or_test', ('and_test (OR and_test)*', reduce_binop)],
    ['test', 'or_test|lambda'],

    ['for_assn_base', 'IDENTIFIER', ('LBRACKET for_assn_list RBRACKET',
        lambda p: p[1])],
    ['for_assn_list', ('for_assn_base (COMMA for_assn_base)*', reduce_list)],
    ['for_assn', ('for_assn_base', lambda p: Target(p[0], info=p.get_info(0)))],
    ['comp_iter', ('FOR for_assn IN test', lambda p: CompIter(p[1], p[3]))],

    # Statements
    ['pass', ('PASS', lambda p: None)],
    ['small_stmt', '(expr_stmt|pass|break|continue|return|yield|import|assert)'],
    ['simple_stmt', ('small_stmt (NEWLINE|SEMICOLON)', lambda p: p[0])],
    ['stmt', ('(simple_stmt|if_stmt|for_stmt|while_stmt|def_stmt|class_stmt) '
        '(NEWLINE|SEMICOLON)*', lambda p: p[0])],
    ['stmt_list', ('stmt*', lambda p: [x for x in p[0] if x is not None])],

    ['break', ('BREAK', lambda p: Break(info=p.get_info(0)))],
    ['continue', ('CONTINUE', lambda p: Continue(info=p.get_info(0)))],
    ['return', ('RETURN test', lambda p: Return(p[1]))],
    ['yield', ('YIELD test', lambda p: Yield(p[1]))],
    ['assert', ('ASSERT test', lambda p: Assert(p[1]))],
]

@libparse.rule_fn(rule_table, 'expr_stmt', 'test [EQUALS test]')
def parse_expr_stmt(p):
    def deconstruct_lhs(lhs):
        if isinstance(lhs, Identifier):
            return lhs.name
        elif isinstance(lhs, List):
            return [deconstruct_lhs(i) for i in lhs]
        lhs.error('invalid lhs for assignment')
    if p[1]:
        return Assignment(Target(deconstruct_lhs(p[0]), info=p.get_info(0)), p[1][1])
    return p[0]

rule_table += [
    # Blocks
    ['delims', ('NEWLINE+', lambda p: None)],
    ['small_stmt_list', ('small_stmt (SEMICOLON small_stmt)*',
        lambda p: [x for x in reduce_list(p) if x is not None])],
    ['block', ('COLON (delims INDENT stmt_list DEDENT|small_stmt_list NEWLINE)',
        lambda p: Block(p[1][2] if len(p[1]) == 4 else p[1][0],
            info=p.get_info(0))),
        ('LBRACE stmt_list RBRACE', lambda p: Block(p[1], info=p.get_info(0)))],
    ['for_stmt', ('FOR for_assn IN test block', lambda p: For(p[1], p[3], p[4]))],
    ['while_stmt', ('WHILE test block', lambda p: While(p[1], p[2]))],
]

@libparse.rule_fn(rule_table,
    'if_stmt', 'IF test block (ELIF test block)* [ELSE block]')
def parse_if_stmt(p):
    else_block = Block([], info=p.get_info(0))
    if p[4]:
        else_block = p[4][1]
    for elif_stmt in reversed(p[3]):
        else_block = IfElse(elif_stmt[1], elif_stmt[2], else_block)
    return IfElse(p[1], p[2], else_block)

# Params
@libparse.rule_fn(rule_table, 'param', 'IDENTIFIER [COLON test] [EQUALS test]')
def parse_param(p):
    return [p[0], p[1][1] if p[1] else None, p[2][1] if p[2] else None]

rule_table += [
    ['param', ('STAR IDENTIFIER', lambda p: StarParams(p[1], info=p.get_info(0)))],
    ['param_list', ('param (COMMA param)*', reduce_list)],
]

@libparse.rule_fn(rule_table, 'params', '[LPAREN [param_list] RPAREN]')
def parse_params(p):
    params, types, starparams, kwparams = [], [], None, []
    if p[0] and p[0][1]:
        for item in p[0][1]:
            if isinstance(item, StarParams):
                assert not starparams
                assert not kwparams
                starparams = item.name
            elif item[2]:
                assert not item[1]
                kwparams.append(KeywordArg(item[0], item[2]))
            else:
                assert not starparams
                assert not kwparams
                params.append(item[0])
                types.append(item[1] or None_(info=NULL_INFO))
    return Params(params, types, starparams, kwparams, info=NULL_INFO)

# Function/class defs
rule_table += [
    ['decorator', ('AT test delims', lambda p: p[1])],
    ['return_type', ('[RARROW test]', lambda p: p[0][1] if p[0] else None)],
    ['lambda', ('LAMBDA params return_type block',
        lambda p: Scope(Function('lambda', p[1], p[2], p[3], info=p.get_info(0))))],
    ['class_stmt', ('CLASS IDENTIFIER params block',
        lambda p: Assignment(Target(p[1], info=p.get_info(1)),
            Scope(Class(p[1], p[2], p[3], info=p.get_info(0)))))],
]

@libparse.rule_fn(rule_table,
    'def_stmt', 'decorator* DEF IDENTIFIER params return_type block')
def parse_def_stmt(p):
    fn = Scope(Function(p[2], p[3], p[4], p[5], info=p.get_info(1)))
    for dec in p[0]:
        fn = Call(dec, [fn])
    return Assignment(Target(p[2], info=p.get_info(2)), fn)

# Imports
def parse_import(p, module, names, path):
    imp = Import([], module, names, path, False, info=p.get_info(0))
    all_imports.append(imp)
    return Scope(imp)

rule_table += [
    ['ident_list', ('IDENTIFIER (COMMA IDENTIFIER)*', reduce_list)],
    ['import', ('IMPORT IDENTIFIER [FROM STRING]',
        lambda p: parse_import(p, p[1], None, p[2][1] if p[2] else None)),
        ('FROM IDENTIFIER IMPORT (ident_list|STAR)',
            lambda p: parse_import(p, p[1], p[3] if p[3] != '*' else [], None))],
]

root_dir = os.path.dirname(sys.path[0])
stdlib_dir = '%s/stdlib' % root_dir

parser = libparse.Parser(rule_table, 'stmt_list')

all_imports = None
module_cache = {}

def parse(path, import_builtins=True, ctx=None):
    global all_imports, filename
    # Check if we've parsed this before. We do a check for recursive imports here too.
    if path in module_cache:
        if module_cache[path] is None:
            raise Exception('recursive import detected for %s' % path)
        return module_cache[path]
    module_cache[path] = None

    # Parse the file
    all_imports = []
    filename = path
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        tokenizer = lexer.Lexer()
        tokenizer.input(f.read(), filename=path)
        block = parser.parse(tokenizer)

    # Do some post-processing, starting with adding builtins
    if import_builtins:
        builtins_path = '%s/__builtins__.mg' % stdlib_dir
        imp = Import([], '__builtins__', [], builtins_path, True, info=builtin_info)
        all_imports.append(imp)
        block = [Scope(imp)] + block

    new_block = []

    for k, v in mg_builtins.builtins.items():
        new_block.append(Assignment(Target(k, info=builtin_info), v))

    # Recursively parse imports. Be sure to copy the all_imports since
    # we'll be clearing and modifying it in each child parsing pass.
    for imp in all_imports[:]:
        # Explicit path: use that
        if imp.path:
            import_paths = [imp.path, '%s/%s' % (dirname, imp.path)]
        else:
            import_paths = ['%s/%s.mg' % (cd, imp.name)
                for cd in [dirname, stdlib_dir]]
        # Normal import: find the file first in
        # the current directory, then stdlib
        for import_path in import_paths:
            if os.path.isfile(import_path):
                break
        else:
            print('checking paths: %s' % import_paths, file=sys.stderr)
            raise Exception('could not find import in path: %s' % imp.name)

        imp.parent_ctx = Context(imp.name, None, ctx)
        imp.stmts = parse(import_path, import_builtins=not imp.is_builtins,
                ctx=imp.parent_ctx)

    new_block.extend(block)

    # Be sure and return a duplicate of the list...
    module_cache[path] = new_block[:]
    return new_block

def interpret(path):
    ctx = Context('__main__', None, None)
    block = parse(path, ctx=ctx)
    preprocess_program(ctx, block)
    try:
        for expr in block:
            expr.eval(ctx)
    except ProgramError as e:
        if e.stack_trace:
            for line in e.stack_trace:
                print(line, file=sys.stderr)
        print(e.msg, file=sys.stderr)
        sys.exit(1)

interpret(sys.argv[1])
