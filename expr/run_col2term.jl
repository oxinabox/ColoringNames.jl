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

const od =(MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.Last())
const serv=SwiftService()

const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_terms_padded, valid_hsv, encoding = prepare_data(valid_raw)

const train_raw = valid_raw
const train_terms_padded = valid_terms_padded
const train_hsv = valid_hsv

#const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
#const train_terms_padded, train_hsv,  encoding = prepare_data(train_raw, encoding)

include("col2term_bigram.jl")

const batch_size = 64_000
n_steps=size(valid_terms_padded,1)-1
n_classes = nlabel(encoding)+1
sess, t = color_to_terms_network(n_classes, n_steps;
        hidden_layer_size = 64,
        embedding_dim = 4,
        batch_size=batch_size;
         )


############################
train_from_terms!(sess, t, train_terms_padded, train_hsv; epochs=50)

(hsv,terms) = eachbatch(
    shuffleobs((train_hsv, train_terms_padded), obsdim=od);
    size=batch_size,
    obsdim=od) |> first

LL,TT = run(sess,
    [t[:LL_masked], t[:TT]],
    Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))

LLim=squeeze(mapslices(indmax, LL, 3),3)-1

target = mapslices(join, ind2label.(max.(terms[2:end-1,:],1)', encoding), 2)
pred = mapslices(join, ind2label.(max.(LLim,1)', encoding), 2)


emt=run(sess, t[:EmbeddingTable])

Xs = run(sess,
    t[:Xs],
    Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))

LLm, TTm = run(sess,
    [t[:LL_flat_masked], t[:TT_flat_masked]],
    Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))
