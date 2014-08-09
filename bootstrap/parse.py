#!/usr/bin/env python3
import os
import sys

import lexer
import sprdpl.lex as liblex
import sprdpl.parse as libparse

import mg_builtins
from syntax import *

# XXX
NULL_INFO = liblex.Info('???')

def reduce_binop(p):
    r = p[0]
    for item in p[1]:
        r = BinaryOp(item[0], r, item[1])
    return r

def reduce_list(p):
    return libparse.ParseResult([p[0]] + [item[1] for item in p[1]],
        [p.info[0]] + [p.info[1][i][1] for i in range(len(p[1]))])

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
        'dict_comp|set_comp'],
]

@libparse.rule_fn(rule_table, 'list_comp', 'LBRACKET [test (comp_iter+|'
    '(COMMA test)* [COMMA])] RBRACKET')
def parse_list(p):
    if p[1]:
        items = p[1]
        l = [items[0]]
        if items[1]:
            if isinstance(items[1][0], CompIter):
                return Scope(ListComprehension(items[0], items[1]))
            l += [item[1] for item in items[1][0]]
        return List(l, info=p.get_info(0))
    return List([], info=p.get_info(0))

@libparse.rule_fn(rule_table, 'dict_comp', 'LBRACE [test COLON test (comp_iter+|'
    '(COMMA test COLON test)* [COMMA])] RBRACE')
def parse_dict(p):
    if p[1]:
        items = p[1]
        key, value = items[0], items[2]
        d = {key: value}
        if items[3]:
            items = items[3]
            if isinstance(items[0], CompIter):
                return Scope(DictComprehension(key, value, items))
            d.update({item[1]: item[3] for item in items[0]})
        return Dict(d, info=p.get_info(0))
    return Dict({}, info=p.get_info(0))

@libparse.rule_fn(rule_table, 'set_comp', 'LBRACE test (COMMA test)* [COMMA] RBRACE')
def parse_set(p):
    set_items = [p[1]] + [p[1] for item in p[2]]
    return Call(Identifier('set', info=p.get_info(0)),
        [List(set_items, info=p.get_info(0))])

@libparse.rule_fn(rule_table, 'arg', 'test [EQUALS test]')
def parse_arg(p):
    if p[1]:
        if not isinstance(p[0], Identifier):
            p.error('invalid keyword argument', 0)
        return KeywordArg(p[0].name, p[1][1])
    return p[0]

@libparse.rule_fn(rule_table, 'subscript', '[test] [COLON [test] [COLON [test]]]]')
def parse_subscript(p):
    [start, stop, step] = [None_(info=NULL_INFO)] * 3
    if p[0]:
        if p[1] is None:
            return lambda expr: GetItem(expr, p[0])
        start = p[0]
    if not p[1]:
        p.error('could not parse subscript')
    if p[1][1] is not None:
        stop = p[1][1]
    if p[1][2] is not None and p[1][2][1] is not None:
        step = p[1][2][1]
    return lambda expr: Call(Identifier('slice', info=NULL_INFO),
        [expr, start, stop, step])

rule_table += [
    # Function calls, subscripts, attribute accesses
    ['vararg', ('STAR test', lambda p: VarArg(p[1]))],
    ['kwvararg', ('STAR_STAR test', lambda p: KeywordVarArg(p[1]))],
    ['args', ('(arg|vararg|kwvararg) (COMMA (arg|vararg|kwvararg))*', reduce_list)],
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
        r = BinaryOp('**', r, p[2][1])
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
    ['mod_setitem', ('LBRACKET test RBRACKET', lambda p: ModItem(p[1], info=p.get_info(0)))],
    ['mod_setattr', ('PERIOD IDENTIFIER', lambda p: ModAttr(p[1], info=p.get_info(0)))],
    ['mod', ('(mod_setitem|mod_setattr)+ EQUALS test', lambda p: ModItems(p[0], p[2]))],
    ['expr', 'mod_expr'],
]

@libparse.rule_fn(rule_table, 'mod_expr', 'or_expr [LARROW mod (COMMA mod)*]')
def parse_mod_expr(p):
    expr = p[0]
    if p[1]:
        return Modification(expr, [p[1][1]] + [item[1] for item in p[1][2]])
    return expr

# XXX chaining comparisons?
@libparse.rule_fn(rule_table, 'comparison', 'expr ((EQUALS_EQUALS|NOT_EQUALS|'
    'GREATER|GREATER_EQUALS|LESS|LESS_EQUALS|IN|NOT IN) expr)*')
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
    ['test', 'lambdef', 'def_expr'],
]

@libparse.rule_fn(rule_table, 'test', 'or_test [IF or_test ELSE test]')
def parse_test(p):
    if p[1]:
        return CondExpr(p[1][1], p[0], p[1][3])
    return p[0]

rule_table += [
    ['for_assn_base', 'IDENTIFIER', ('LBRACKET for_assn_list RBRACKET',
        lambda p: p[1])],
    ['for_assn_list', ('for_assn_base (COMMA for_assn_base)*', reduce_list)],
    ['for_assn', ('for_assn_base', lambda p: Target([p[0]], info=NULL_INFO))],
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

@libparse.rule_fn(rule_table, 'expr_stmt', 'test (EQUALS test)*')
def parse_expr_stmt(p):
    def deconstruct_lhs(lhs, p, index):
        if isinstance(lhs, Identifier):
            return lhs.name
        elif isinstance(lhs, List):
            return [deconstruct_lhs(item, p, index) for i, item in enumerate(lhs)]
        p.error('invalid lhs for assignment', index)
    # XXX HACK
    r = reduce_list(p)
    all_items = r.items
    [targets, expr] = [all_items[:-1], all_items[-1]]
    if targets:
        return Assignment(Target([deconstruct_lhs(t, r, i) for i, t in enumerate(targets)],
            info=NULL_INFO), expr)
    return expr

rule_table += [
    # Blocks
    ['delims', ('NEWLINE+', lambda p: None)],
    ['small_stmt_list', ('small_stmt (SEMICOLON small_stmt)*',
        lambda p: [x for x in reduce_list(p).items if x is not None])],
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
    else_block = p[4][1] if p[4] else Block([], info=p.get_info(0))
    for elif_stmt in reversed(p[3]):
        else_block = IfElse(elif_stmt[1], elif_stmt[2], else_block)
    return IfElse(p[1], p[2], else_block)

# Params
@libparse.rule_fn(rule_table, 'param', 'IDENTIFIER [COLON test] [EQUALS test]')
def parse_param(p):
    return [p[0], p[1][1] if p[1] else None, p[2][1] if p[2] else None]

rule_table += [
    ['param', ('STAR IDENTIFIER', lambda p: VarParams(p[1], info=p.get_info(0))),
        ('STAR_STAR IDENTIFIER', lambda p: KeywordVarParams(p[1], info=p.get_info(0)))],
    ['param_list', ('param (COMMA param)*', reduce_list)],
]

@libparse.rule_fn(rule_table, 'params', '[LPAREN [param_list] RPAREN]')
def parse_params(p):
    params, types, var_params, kwparams, kw_var_params = [], [], None, [], None
    if p[0] and p[0][1]:
        for i, item in enumerate(p[0][1]):
            indices = [0, 1, i]
            if isinstance(item, VarParams):
                if var_params:
                    p.error('only one varparam (*) allowed', *indices)
                if kwparams or kw_var_params:
                    p.error('a varparam (*) must come before keyword parameters', *indices)
                var_params = item.name
            elif isinstance(item, KeywordVarParams):
                kw_var_params = item.name
            elif item[2]:
                if kw_var_params:
                    p.error('keyword parameters cannot come after keyword varparams (**)', *indices)
                if item[1]:
                    p.error('typed keyword parameters are not currently supported', *indices)
                kwparams.append(KeywordArg(item[0], item[2]))
            else:
                if var_params or kwparams or kw_var_params:
                    p.error('positional arguments must appear before varparams or keyword parameters', *indices)
                params.append(item[0])
                types.append(item[1] or None_(info=NULL_INFO))
    return Params(params, types, var_params, kwparams, kw_var_params,
        info=NULL_INFO)

# Function/class defs
rule_table += [
    ['decorator', ('AT test (NEWLINE|SEMICOLON)+', lambda p: p[1])],
    ['return_type', ('[RARROW test]', lambda p: p[0][1] if p[0] else None)],
    ['lambdef', ('LAMBDA params COLON test',
        lambda p: Scope(Function('lambda', p[1], None,
            Block([Return(p[3])], info=p.get_info(0)), info=p.get_info(0))))],
    ['def_expr', ('DEF params return_type block',
        lambda p: Scope(Function('lambda', p[1], p[2], p[3], info=p.get_info(0))))],
    ['class_stmt', ('CLASS IDENTIFIER params block',
        lambda p: Assignment(Target([p[1]], info=p.get_info(1)),
            Scope(Class(p[1], p[2], p[3], info=p.get_info(0)))))],
]

@libparse.rule_fn(rule_table,
    'def_stmt', 'decorator* DEF IDENTIFIER params return_type block')
def parse_def_stmt(p):
    fn = Scope(Function(p[2], p[3], p[4], p[5], info=p.get_info(1)))
    for dec in p[0]:
        fn = Call(dec, [fn])
    return Assignment(Target([p[2]], info=p.get_info(2)), fn)

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
tokenizer = lexer.Lexer()

all_imports = None
module_cache = {}

def parse(path, import_builtins=True, ctx=None):
    global all_imports
    # Check if we've parsed this before. We do a check for recursive imports here too.
    if path in module_cache:
        if module_cache[path] is None:
            raise Exception('recursive import detected for %s' % path)
        return module_cache[path]
    module_cache[path] = None

    # Parse the file
    all_imports = []
    dirname = os.path.dirname(path)
    if not dirname:
        dirname = '.'
    with open(path) as f:
        block = parser.parse(tokenizer.input(f.read(), filename=path))

    # Do some post-processing, starting with adding builtins
    if import_builtins:
        builtins_path = '%s/__builtins__.mg' % stdlib_dir
        imp = Import([], '__builtins__', [], builtins_path, True, info=builtin_info)
        all_imports.append(imp)
        block = [Scope(imp)] + block

    new_block = []

    for k, v in mg_builtins.builtins.items():
        new_block.append(Assignment(Target([k], info=builtin_info), v))

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

def interpret(path, print_program=False):
    ctx = Context('__main__', None, None)
    block = parse(path, ctx=ctx)
    preprocess_program(ctx, block)
    try:
        if print_program:
            for expr in block:
                print(expr.str(ctx))
        for expr in block:
            expr.eval(ctx)
    except ProgramError as e:
        if e.stack_trace:
            for line in e.stack_trace:
                print(line, file=sys.stderr)
        print(e.msg, file=sys.stderr)
        sys.exit(1)

def main(args):
    print_program = False
    if args[1] == '--print':
        print_program = True
        args.pop(1)

    interpret(args[1], print_program=print_program)

if __name__ == '__main__':
    main(sys.argv)
