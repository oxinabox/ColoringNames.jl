
using TensorFlow
using Distributions
using MLDataUtils
using SwiftObjectStores
using Iterators
using ColoringNames
using MLLabelUtils
using StaticArrays
using ProgressMeter
using StatsBase


const od =(MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.Last())

function prepare_data(raw, encoding_=nothing, tokenize=morpheme_tokenize)
    labels = convert(Vector{String}, raw[:,1]);
    hsv_data = convert(Matrix{Float64}, raw[:,2:end]);
    tokenized_labels = demarcate.(tokenize.(labels))
    local encoding
    if encoding_===nothing
        all_tokens = reduce(union, tokenized_labels)
        encoding = labelenc(all_tokens)
    else
        encoding = encoding_
    end

    label_inds = map(toks->label2ind.(toks, Scalar(encoding)), tokenized_labels)
    rpad_to_matrix(label_inds), hsv_data, encoding
end


get_mask(V, dtype=Float32)=cast(V, Bool)
apply_mask(V, mask) = gather_nd(V, find(mask))
#V.*tile(expand_dims(mask, 1), [1, get_shape(V,2)])

#DEFINITION
function color_to_terms_network(n_classes, n_steps;
        batch_size = 128,
        learning_rate = 0.05,
        hidden_layer_size = 256,
        embedding_dim = 16
    )
    n_input = 3 # HSV


    ###################################
    sess = Session(Graph())
    tensor_vars = @names_from begin
        X_hsv = placeholder(Float32, shape=[batch_size, n_input]; name="X_HSVs")
        Term_obs_s = placeholder(Int32, shape=[n_steps+1, batch_size]; name="Term_obs_s")

        EmbeddingTable = get_variable("TokenEmbeddings3",  [n_classes, embedding_dim], Float32; initializer=Normal(0, .1))


        #Mangle Terms into shape
        Term_obs_s_out = slice(Term_obs_s, [1,0], [-1,-1]) #Don't want first input "<S>" #0based
        TT = reshape(Term_obs_s_out, [n_steps*batch_size]; name="Stack_Term_obs_s")

        Term_obs_s_ins = unpack(Term_obs_s+1)[1:end-1]#Don't want last input "</S>" (or padding character often but we will handle that seperately)
        Tes = gather.(Scalar(EmbeddingTable), Term_obs_s_ins)

        @show get_shape.(Tes)


        #Mangle colors into shape
        X_h, X_s, X_v = unpack(X_hsv; axis=2)
        X_h = reshape(X_h, [batch_size])
        X_hr = X_h.*2π
        X_col = pack((sin(X_hr), cos(X_hr), X_s-0.5, X_v-0.5); axis=2) #Smooth hue by breaking into cos and sin, and zero mean everything else1
        Xs = [concat(2, [X_col, T]; name="Xs$ii") for (ii,T) in enumerate(Tes)]#Pair color input at each step with previous term


        Hs, states = nn.rnn(nn.rnn_cell.LSTMCell(hidden_layer_size), Xs; dtype=Float32)#, sequence_length=n_steps);
        W1 = get_variable("weights2", [hidden_layer_size, n_classes], Float32;  initializer=Normal(0, .1))
        B1 = get_variable("bias2", [n_classes], Float32;  initializer=Normal(0, .1))
        Ls =  [H*W1+B1 for H in Hs]


        LL = concat(1, Ls; name="Stack_Logits")

        mask = get_mask(TT)
        TT_masked = apply_mask(TT, mask)
        LL_masked = apply_mask(LL, mask)
        @show get_shape(find(mask))
        costs = nn.sparse_softmax_cross_entropy_with_logits(LL_masked, TT_masked+1)
        cost = reduce_mean(-costs) #cross entropy
        optimizer = train.minimize(train.AdamOptimizer(learning_rate), cost)


        Term_preds_onehots = nn.softmax(LL; name="Term_preds_onehots")
        Term_preds_s = reshape(indmax(Term_preds_onehots, 2)+1, [n_steps, batch_size]) #TODO: this messes up zero entries, not that it matters

        #costs  = reduce_sum(Term_obs_onehots.*Term_preds_onehots_log, reduction_indices=[1])


    end
    ########## GET it running

    run(sess, initialize_all_variables())

    return sess, tensor_vars
end



function train_from_terms!(sess, t::Associative{Symbol}, train_terms_padded, train_hsv; epochs=3)
    batchs = eachbatch(shuffleobs((train_hsv, train_terms_padded); obsdim=od); size=batch_size, obsdim=od)
    local cost_o
    for ii in 1:epochs
        @show ii
        @showprogress for (hsv,terms) in batchs
            Term_preds_s_o, TT_s_o, cost_o, optimizer_o = run(sess,
                [t[:Term_preds_s], t[:TT], t[:cost], t[:optimizer]],
            Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))
        end
    end
    @show cost_o
end

"A quick hack to calculate accuracy"
function masked_acc(obs, pred)
    obs = obs[2:end,:] #skip start token
    @assert(size(obs)==size(pred), "$(size(obs)) != $(size(pred))")
    mpred = sign(obs).*pred
    mean(pp==oo for (pp,oo) in eachobs((mpred, obs)))
end

"A quick hack to calculate perplexity"
function masked_perp(obs, preds_onehots_log)
    obs = obs[2:end,:] #skip start token
    pred_lp=reshape(preds_onehots_log, (batch_size, n_steps, n_classes))
    expon = 0.0
    for (o_indexes, log_probs) in eachobs((obs, pred_lp),
        obsdim=(MLDataUtils.ObsDim.Last(), MLDataUtils.ObsDim.First()))
        content_o_indexes = rstrip(o_indexes)[end-1] #strip padding and end token
        expon += sum(log_probs[step, ii] for (step,ii) in enumerate(content_o_indexes))
        # Each probability is conditional on all of the ealier.
        # Their product is the competely correct (*without* Markov assumption)
        # probability estimate for the whole
        # Adding the Logs is thus a correct log-prob for the whole
    end
    exp(-expon/batch_size)
end

"""
This evaluation function does not take into account the non-knowing of term t-1.
This is legal for calculating perplexity
"""
function rough_evalute(sess, t::Associative{Symbol}, test_terms_padded, test_hsv)
    batchs = eachbatch((test_hsv, test_terms_padded); size=batch_size, obsdim=od)
    cost_total = 0.0
    acc_total = 0.0
    perps = Float64[]
    preds_o = Matrix{Float64}(size(test_terms_padded,1)-1,0)
    @showprogress for (hsv,terms) in batchs
        preds, preds_lp, cost_o = run(sess,
        [t[:Term_preds_s], t[:Term_preds_onehots_log], t[:cost]],
        Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))

        preds_o = [preds_o preds]

        push!(perps, masked_perp(terms, preds_lp))
        acc_total += masked_acc(terms, preds)
        cost_total += cost_o
    end

    (cost_total/length(batchs),
    acc_total/length(batchs),
    geomean(perps),
    preds_o)
end
