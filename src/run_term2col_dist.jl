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


include("term2col_dist.jl")

batch_size = 64_000
output_res = 64
n_steps=size(valid_terms_padded,1)
n_classes = nlabel(encoding)+1

sess, optimizer = terms_to_color_dist_network(n_classes, n_steps;
                                            output_res= output_res,
                                            batch_size = batch_size,
                                            hidden_layer_size = 256,
                                            learning_rate = 0.5)
costs_o = train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsv; epochs=30)
#######################

data = shuffleobs((train_hsv, train_terms_padded); obsdim=od)
hsv, terms = eachbatch(data; size=batch_size, obsdim=od) |> first
hp_obs, sp_obs, vp_obs = splay_probabilities(hsv, output_res)

ss=sess.graph
hp, sp, vp = run(
    sess,
    [
        ss["Yp_hue"],
        ss["Yp_sat"],
        ss["Yp_val"]
    ],
    Dict(
        ss["terms"]=>terms,
        ss["Yp_obs_hue"]=>hp_obs,
        ss["Yp_obs_sat"]=>sp_obs,
        ss["Yp_obs_val"]=>vp_obs
    )
)


using Plots
gr()

ii=103
bar(
    [hp[ii,:] hp_obs[:,ii] sp[ii,:] sp_obs[:,ii] vp[ii,:] vp_obs[:,ii]],
    legend = false,
    #title = ["h", "h_obs", "s", "s_obs", "v_obs"],
    layout=(3,2))
















'|
