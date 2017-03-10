module ColoringNames

using MLDataUtils
using Iterators
using Memoize

export morpheme_tokenizer, morpheme_tokenize, demarcate, rpad_to_matrix, @names_from

include("util.jl")

include("input.jl")



end # module
