module ColoringNames
using MLLabelUtils
using MLDataUtils
using Iterators
using Memoize
using StaticArrays

export morpheme_tokenizer, morpheme_tokenize, demarcate, rpad_to_matrix, @names_from, prepare_data

include("util.jl")

include("input.jl")



end # module
