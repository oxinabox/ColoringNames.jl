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

include("term2col.jl")

const batch_size = 64_000
n_steps = size(valid_terms_padded,1)
n_classes = nlabel(encoding)+1
sess, opt = terms_to_color_network(n_classes, n_steps;
        hidden_layer_size = 64,
        embedding_dim = 16,
        batch_size=batch_size; )
ss = sess.graph

############################

ss = sess.graph
costs_o = Float64[]
@progress "Epochs" for ii in 1:100
    @show ii
    data = shuffleobs((train_hsv, train_terms_padded); obsdim=od)
    #data = undersample((train_hsv, train_terms_padded); obsdim=od, shuffleobs=true)
    batchs = eachbatch(data; size=batch_size, obsdim=od)
    @progress "Batches" for (hsv,terms) in batchs
        cost_o, optimizer_o = run(sess,
            [ss["cost"], opt],
        Dict(ss["Yhuesatval_obs"]=>hsv, ss["terms"]=>terms))
        push!(costs_o, cost_o)
    end
end
mean(costs_o)


od_with_text = (od..., LearnBase.ObsDim.Last())
(hsv_obs, terms, text_obs) = eachbatch(
    #shuffleobs((train_hsv, train_terms_padded, train_text), obsdim=od_with_text);
    (train_hsv, train_terms_padded, train_text),
    size=batch_size,
    obsdim=od_with_text) |> first

hsv_pred = run(sess, ss["Yhuesatval"], Dict(ss["terms"]=>terms))


col_pred = mapslices(x->RGB(HSV(360*x[1], x[2], x[3])), hsv_pred, 2)
col_obs = mapslices(x->RGB(HSV(360*x[1], x[2], x[3])), hsv_obs, 2)

show_idx=rand(1:batch_size, 20)
plot_colors(col_pred[show_idx], col_obs[show_idx];
            row_names=text_obs[show_idx], column_names=["pred", "obs"])
