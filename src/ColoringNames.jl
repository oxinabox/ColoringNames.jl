__precompile__(true)
module ColoringNames
using IterTools
using Memoize
using TensorFlow
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

export plot_colors, plot_hsv


export morpheme_tokenizer, morpheme_tokenize, demarcate, rpad_to_matrix, @names_from, prepare_data, prepare_labels, hsv2colorant
export descretized_perplexity, find_bin, mse_from_peak, peak, bin_expected_value, total_descretized_logprob
export load_munroe_data, rare_descriptions, ColorDatasets, ColorDataset, extrapolation_dataset
export stringify_keys
export TermToColorDistributionSOWE
export find_distributions, load_word2vec_embeddings, load_text_embeddings, load_color_nameset
export output_res


include("init_datadeps.jl")

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
include("networks/term2col_dist.jl")
include("networks/term2col_dist_noml.jl")
include("networks/term2col_dist_SOWE.jl")

end # module
