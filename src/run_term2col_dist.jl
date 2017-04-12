using SwiftObjectStores
using ColoringNames
using TensorFlow
using Distributions
using MLDataUtils
using Iterators
using MLLabelUtils
using StaticArrays
using Juno
using StatsBase
using Colors


const od =(MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.Last())
const serv=SwiftService()

const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw; do_demacate=false)
const valid_text = valid_raw[:, 1]

const train_raw = valid_raw
const train_terms_padded = valid_terms_padded
const train_hsv = valid_hsv
const train_text = valid_text
#const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
#const train_hsv, train_terms_padded, encoding = prepare_data(train_raw; do_demacate=false)

function splay_probabilities(hsv, nbins, stddev=1/nbins)
    num_obs =  nobs(hsv, ObsDim.First())
    hp = Matrix{Float32}((nbins, num_obs))
    sp = Matrix{Float32}((nbins, num_obs))
    vp = Matrix{Float32}((nbins, num_obs))
    @progress for (ii, obs) in enumerate(eachobs(hsv, ObsDim.First()))
        #GOLDPLATE: Make this nonallocating
        hp[:,ii] = vonmiseshot(hsv[1], nbins, stddev)
        sp[:,ii] = gaussianhot(hsv[2], nbins, stddev)
        vp[:,ii] = gaussianhot(hsv[3], nbins, stddev)
    end
    (hp, sp, vp)
end

splay_probabilities(valid_hsv, 256)

include("term2col_dist.jl")
