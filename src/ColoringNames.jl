module ColoringNames
using MLLabelUtils
using MLDataUtils
using Iterators
using Memoize
using StaticArrays
using TensorFlow

export morpheme_tokenizer, morpheme_tokenize, demarcate, rpad_to_matrix, @names_from, prepare_data

include("util.jl")
include("tf_extra.jl")

include("input.jl")





end # module
