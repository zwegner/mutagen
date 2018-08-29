" Vim syntax file for mutagen

runtime! syntax/python.vim

let b:current_syntax = "mutagen"
let python_highlight_all = 1

syn keyword mutagenStatement union nextgroup=mutagenType
syn keyword mutagenConditional match
syn keyword mutagenConditional consume
syn keyword mutagenConditional effect
syn keyword mutagenConditional perform
syn keyword mutagenConditional resume

syn match mutagenType "[^:([]*" contained

hi link mutagenStatement Statement
hi link mutagenConditional pythonConditional
hi link mutagenType Function
