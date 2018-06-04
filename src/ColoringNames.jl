__precompile__(false)
module ColoringNames
using TensorFlow
import_op("Atan2")

using IterTools
using Memoize
using StatsBase
using DataDeps
import Juno: @progress
using Distributions
using CatViews
using Base.Threads
using DataStructures
using MLDataUtils
using StaticArrays
import NamedColors
using FileIO, JLD
using Plots
using KernelDensity

export plot_colors, plot_hsv, plot_query
export morpheme_tokenize, rpad_to_matrix, @names_from, prepare_data, prepare_labels, hsv2colorant
export descretized_perplexity, find_bin, mse_from_peak, mse_from_distmean, distmean, peak, bin_expected_value, total_descretized_logprob
export load_munroe_data, rare_descriptions, ColorDatasets, ColorDataset, extrapolation_dataset, order_relevant_dataset, order_relevant_name_pairs
export stringify_keys, laplace_smooth

export TermToColorDistributionSOWE, TermToColorDistributionRNN, TermToColorDistributionCNN, TermToColorDistributionEmpirical
export TermToColorPointSOWE, TermToColorPointRNN, TermToColorPointCNN, TermToColorPointEmpirical

export find_distributions, load_word2vec_embeddings, load_text_embeddings, load_color_nameset
export train!, query, evaluate, restore
export output_res


include("init_datadeps.jl")


abstract type AbstractDistEstModel end
abstract type AbstractPointEstModel end

abstract type AbstractDistEstML  <: AbstractDistEstModel end
abstract type AbstractPointEstML <: AbstractPointEstModel end
const AbstractModelML = Union{AbstractDistEstML, AbstractPointEstML}


include("util.jl")
include("tf_extra.jl")

include("input.jl")
include("load_data.jl")

include("embeddings.jl")

include("prob.jl")
include("prob2.jl")

include("plot_colors.jl")

include("evaluation.jl")

include("networks/networks_common.jl")
include("networks/term2col_noml.jl")
include("networks/term2col_SOWE.jl")
include("networks/term2col_RNN.jl")
include("networks/term2col_CNN.jl")

end # module
