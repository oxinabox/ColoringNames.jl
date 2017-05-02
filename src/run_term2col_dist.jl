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

batch_size = size(valid_terms_padded,2)
output_res = 64
n_steps=size(valid_terms_padded,1)
n_classes = nlabel(encoding)+1

sess, optimizer = terms_to_color_dist_network(n_classes, n_steps;
                                            output_res= output_res,
                                            batch_size = batch_size,
                                            embedding_dim = 32,
                                            hidden_layer_size = 256,
                                            learning_rate = 0.5)
costs_o = train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsv; epochs=200)


using Plots
gr()
plot(1:length(costs_o), costs_o)

#### Save It
saver = train.Saver()
save_path = joinpath(Pkg.dir("ColoringNames"), "models", "1.jld")
train.save(saver, sess, save_path)

sess2, _ = terms_to_color_dist_network(n_classes, n_steps;
                                            output_res= output_res,
                                            batch_size = batch_size,
                                            embedding_dim = 32,
                                            hidden_layer_size = 256,
                                            learning_rate = 0.5)
saver2 = train.Saver()
train.restore(saver2, sess2, save_path)

#######################
# Lets look at the output


##
#
function querier(sess, batch_size, n_steps; encoding=encoding)
    function query(input_text)
        label = SubString(input_text, 1) #HACK to get String->SubString
        labels, _ = ColoringNames.prepare_labels([label], encoding, do_demacate=false)

        nsteps_to_pad = n_steps - size(labels,1)
        nbatch_items_to_pad = batch_size - size(labels,2)

        padded_labels = [[labels; zeros(Int, nsteps_to_pad)] zeros(Int, (n_steps, nbatch_items_to_pad))]
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
                ss["terms"]=>padded_labels,
            )
        )

        hp[1,:], sp[1,:], vp[1,:]
    end
end

const query = querier(sess, batch_size, n_steps; encoding=encoding)
pyplot()

plot_query(input) = plot_hsv(query(input)...)


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

plot_query("yellow ish blue")
plot_query("yellow y blue")

plot_query("green ish blue")
plot_query("green y blue")
plot_query("blue ish green")
plot_query("blue ish green")

plot_query("pale blue")
plot_query("pale ish blue")
plot_query("dark ish blue")
plot_query("dark blue")

plot_query("vomit")
plot_query("dark vomit")
plot_query("bright vomit")

plot_query("bright grey")
plot_query("grey")
plot_query("dark grey")

plot_query("hot pink")
plot_query("hot green")
plot_query("acid green")
plot_query("acid pink")

plot_query("pale hot pink")
plot_query("very very hot pink") #fails

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

#Check correlations
cor(hsv[same_terms,1], hsv[same_terms,2])
cor(hsv[same_terms,1], hsv[same_terms,3])
cor(hsv[same_terms,3], hsv[same_terms,2])

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























plot_hsv(hp::Vector, sp::Vector, vp::Vector)
    hp, sp, vp = query(input)
    nbins = length(hp)
    @assert nbins == length(sp) == length(vp)
    h_max, s_max, v_max = (indmax.([hp, sp, vp]))/nbins
    @show h_max, s_max, v_max
    h_bar_colors = ColoringNames.hsv2colorant([linspace(0.0,1.0, nbins) s_max*ones(nbins) v_max*ones(nbins)])
    s_bar_colors = ColoringNames.hsv2colorant([h_max*ones(nbins) linspace(0.0,1.0, nbins) v_max*ones(nbins)])
    v_bar_colors = ColoringNames.hsv2colorant([h_max*ones(nbins) s_max*ones(nbins) linspace(0.0,1.0, nbins)])
    #
    bar([hp, sp, vp], legend = false, layout=(1,3), linewidth=0, seriescolor=[h_bar_colors s_bar_colors v_bar_colors])
end

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

plot_query("pale blue")
plot_query("pale ish blue")
plot_query("dark ish blue")
plot_query("dark blue")


bar([1,2,3], fillcolor=[:red, :green, :blue])






plot_query("hot pink")
plot_query("hot green")
plot_query("acid green")
plot_query("acid pink")

plot_query("pale hot pink")
plot_query("very very hot pink")

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

same_terms = same_term_indexes_set[21]
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




#EOF
