module ColoringNames
using MLLabelUtils
using MLDataPattern
using Iterators
using Memoize
using StaticArrays
using TensorFlow
using StatsBase

export morpheme_tokenizer, morpheme_tokenize, demarcate, rpad_to_matrix, @names_from, prepare_data, prepare_labels, hsv2colorant


"Converts and array 3 values between 0 and 1 for HSV to a colorant"
hsv2colorant(x1, x2, x3) = RGB(HSV(360*x1, x2, x3))

hsv2colorant(x::AbstractVector) = hsv2colorant(x...)
function hsv2colorant(hsvs::AbstractMatrix)
    @assert size(hsvs, 2) == 3
    mapslices(hsv2colorant, hsvs, 2)
 end


include("util.jl")
include("tf_extra.jl")

include("input.jl")
include("load_data.jl")

include("prob.jl")
include("plot_colors.jl")

include("evaluation.jl")

include("networks/term2col_dist.jl")
include("networks/term2col_dist_noml.jl")
end # module
