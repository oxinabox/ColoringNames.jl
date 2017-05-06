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

const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
const train_hsv, train_terms_padded, encoding = prepare_data(train_raw, encoding; do_demacate=false)
const train_text = train_raw[:, 1]

include("term2col_dist.jl")



run_data = @names_from begin
    save_path = joinpath(Pkg.dir("ColoringNames"), "models", "1.jld")
    log_path = joinpath(Pkg.dir("ColoringNames"), "logs")

    batch_size = size(valid_terms_padded,2)
    output_res = 64
    n_steps=size(valid_terms_padded,1)
    n_classes = nlabel(encoding)+1
    hidden_layer_size = 256
    embedding_dim = 32
    learning_rate = 0.5
end

## Train
sess, optimizer = terms_to_color_dist_network(n_classes, n_steps;
                                            output_res= output_res,
                                            batch_size = batch_size,
                                            embedding_dim = embedding_dim,
                                            hidden_layer_size = hidden_layer_size,
                                            learning_rate = learning_rate)

costs_o = train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsv; epochs=200)



#### Save It
train.save(train.Saver(), sess, save_path)

#### Load it
sess, _ = terms_to_color_dist_network(n_classes, n_steps;
                                        output_res= output_res,
                                        batch_size = batch_size,
                                        embedding_dim = embedding_dim,
                                        hidden_layer_size = hidden_layer_size,
                                        learning_rate = learning_rate)

train.restore(train.Saver(), sess, save_path)

summary_writer = train.SummaryWriter(log_path)
write(summary_writer, sess.graph)
close(summary_writer)
#########
# calculate binned-perplexity


gg=sess.graph
Yp_hue, Yp_sat, Yp_val = run(sess, [gg["Yp_hue"], gg["Yp_sat"], gg["Yp_val"]],  Dict(gg["terms"]=>valid_terms_padded, gg["keep_prob"]=>1.0))

Y_obs_hue = valid_hsv[:, 1]
Y_obs_sat = valid_hsv[:, 2]
Y_obs_val = valid_hsv[:, 3]

perp_hue = descretized_perplexity(Y_obs_hue, Yp_hue)
perp_sat = descretized_perplexity(Y_obs_sat, Yp_sat)
perp_val = descretized_perplexity(Y_obs_val, Yp_val)
perp = geomean([perp_hue perp_sat perp_val])

Yp_uniform = ones(length(Y_obs_hue), output_res)./length(Y_obs_hue)
perp_uniform_baseline = descretized_perplexity(Y_obs_hue, Yp_uniform)


####
# calcute MSE from the peak

mse_from_peak(valid_hsv, (Yp_hue, Yp_sat, Yp_val))

#######################
# Lets look at the output

const query = querier(sess, batch_size, n_steps; encoding=encoding)
using Plots
pyplot()

plot_query(input) = plot_hsv(query(input)...)

##
#



plot_query("greenish blue")
plot_query("very pale pink")
plot_query("pink")

plot_query("very very pale blue")
plot_query("very pale blue")
plot_query("pale blue")
plot_query("blue")
plot_query("dark blue")
plot_query("very dark blue")
plot_query("very very dark blue")



######


#data = shuffleobs((train_hsv, train_terms_padded); obsdim=od)
data = (valid_hsv, valid_terms_padded)
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
        ss["keep_prob"]=>1.0f0,
        ss["terms"]=>terms,
        ss["Yp_obs_hue"]=>hp_obs,
        ss["Yp_obs_sat"]=>sp_obs,
        ss["Yp_obs_val"]=>vp_obs
    )
)



ii=5
bar(
    [hp[ii,:] hp_obs[:,ii] sp[ii,:] sp_obs[:,ii] vp[ii,:] vp_obs[:,ii]],
    legend = false,
    #title = ["h", "h_obs", "s", "s_obs", "v_obs"],
    layout=(3,2))

#expected output
same_term_indexes_set = map(x->first.(x), groupby(kv->kv[2], enumerate(obsview(terms))))



same_terms = same_term_indexes_set[99]
@show valid_text[first(same_terms)]

bar(
    [hp[first(same_terms),:],sp[first(same_terms),:], vp[first(same_terms),:],
     mean(hp_obs[:,same_terms], 2), mean(sp_obs[:,same_terms], 2), mean(vp_obs[:,same_terms], 2)
    ],
    legend = false,
    layout = (2,3)
    )

##############
# Check Biases
using StatsFuns

hb, sb, vb = run(
    sess,
    [
        ss["B_hue"],
        ss["B_sat"],
        ss["B_val"]
    ]
)


bar(
    softmax.([hb, sb, vb]),
    legend = false,
    #title = ["h", "h_obs", "s", "s_obs", "v_obs"],
    layout=(3,1))



#######################
# Check Embeddings
using MultivariateStats
using TSne
emb_table = run(sess, ss["emb_table"])[2:end, :]' #Skip the first column -- that is padding token

embs_dr = transform(fit(PCA, emb_table; maxoutdim=2), emb_table)
embs_dr = tsne(emb_table', 3, 100, 1000, 20.0)'

scatter(embs_dr[1,:], embs_dr[2,:], series_annotations = encoding.label, legend=false)



using NearestNeighbors

nntree = KDTree(emb_table)

function lookup(input_word, k=10)
    input_word_x = split(input_word) |> first #HACK to deal with Substring != String
    @assert length(input_word)==length(input_word_x)
    input_id = label2ind(input_word_x, encoding)
    point = emb_table[:, input_id]
    idxs, dists = knn(nntree, point, k, true)
    [ind2label.(idxs, encoding) dists]
end

lookup("crimson")



#EOF
