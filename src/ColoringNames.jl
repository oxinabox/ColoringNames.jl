__precompile__(false)
module ColoringNames
using MLLabelUtils
using MLDataPattern
using IterTools
using Memoize
using StaticArrays
using TensorFlow
using StatsBase
using DataDeps
import Juno: @progress
using Distributions
using CatViews
using Base.Threads

export morpheme_tokenizer, morpheme_tokenize, demarcate, rpad_to_matrix, @names_from, prepare_data, prepare_labels, hsv2colorant
export find_distributions

include("init_datadeps.jl")

include("util.jl")
include("tf_extra.jl")

include("input.jl")
include("load_data.jl")

include("prob.jl")
include("prob2.jl")

include("plot_colors.jl")

include("evaluation.jl")

include("networks/term2col_dist.jl")
include("networks/term2col_dist_noml.jl")
include("networks_common.jl")
end # module
