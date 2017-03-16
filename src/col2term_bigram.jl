



get_mask(V)=cast(V, Bool)
apply_mask(V, mask) = gather_nd(V, find(mask))
unwrap_mask(masked_vals, mask, original_vals) =  scatter_nd(find(mask), masked_vals, size(original_vals))


"""
Does a Matrix multiplication on of the final dimention of a tensor with the Matrix
Equivelent  to in julia `(A,B) -> mapslices(Ā->Ā*B, B, 2:3)` (for 3D A)
"""
function trailing_matmul(A,B)
    A_dims = size(A)
    B_dims = size(B)
    A_flat_dims = reduce_prod(A_dims[1:end-1])

    Af = reshape(A, stack([A_flat_dims, A_dims[end]]))
    ABf = Af*B
    AB = reshape(ABf, concat([A_dims[1:end-1], expand_dims(B_dims[end], 1)],1))
    AB
end

#DEFINITION
function color_to_terms_network(n_classes, n_steps;
        batch_size = 128,
        hidden_layer_size = 256,
        embedding_dim = 16,
        learning_rate=0.05
    )
    n_input = 3 # HSV


    ###################################
    sess = Session(Graph())
    tensor_vars = @names_from begin

        X_hsv = placeholder(Float32, shape=[batch_size, n_input]; name="X_HSVs")
        Term_obs_s = placeholder(Int32, shape=[n_steps+1, batch_size]; name="Term_obs_s")

        EmbeddingTable = one_hot(collect(1:314), 314)
        #EmbeddingTable = get_variable("TokenEmbeddings", [n_classes, embedding_dim], Float32; initializer=Normal(0, .01), trainable=true)


        #Mangle Terms into shape
        Term_obs = unstack(Term_obs_s)
        TT = stack(Term_obs[2:end]) #Skip first input which will be <S>

        Term_obs_s_ins = Term_obs[1:end-1]#Don't want last input "</S>" (or padding character often but we will handle that seperately)

        Tes = [gather(EmbeddingTable, term+1) for term in Term_obs_s_ins] #+1 because Gather is 1 indexed


        #Mangle colors into shape
        X_h, X_s, X_v = unstack(X_hsv; axis=2)
        #X_h = reshape(X_h, [batch_size])
        X_hr = X_h.*2π
        X_col = stack((sin(X_hr), cos(X_hr), X_s-0.5, X_v-0.5); axis=2) #Smooth hue by breaking into cos and sin, and zero mean everything else1
        Xs = [concat([X_col, T], 2; name="Xs$ii") for (ii,T) in enumerate(Tes)]# Pair color input at each step with previous term



        @show get_shape.(Xs)
        #cell = nn.rnn_cell.LSTMCell(hidden_layer_size)
        #H1s, states = nn.rnn(cell, Xs; dtype=Float32)
        #@show cell
        #@show H1s
        #@show states
        W0 = get_variable("weights0", [get_shape(Xs[1],2), hidden_layer_size], Float32;  initializer=Normal(0, .01))
        B0 = get_variable("bias0", [hidden_layer_size], Float32;  initializer=Normal(0, .01))
        H1s = [nn.sigmoid(X*W0+B0) for X in Xs]
        @show
        W1 = get_variable("weights1", [hidden_layer_size, n_classes], Float32;  initializer=Normal(0, .01))
        B1 = get_variable("bias1", [n_classes], Float32;  initializer=Normal(0, .01))
        LL =  stack([H*W1+B1 for H in H1s])

        mask = (TT .!= Int32(0)) &  (TT .!= Int32(4))
        TT_flat_masked = apply_mask(TT, mask)
        LL_flat_masked = apply_mask(LL, mask)

        TT_masked = unwrap_mask(TT_flat_masked, mask, TT)
        LL_masked = unwrap_mask(LL_flat_masked, mask, LL)

        acc_costs = nn.sparse_softmax_cross_entropy_with_logits(logits=LL_flat_masked, labels=TT_flat_masked+1) #Add one as TT is 0 based
        cost = reduce_mean(acc_costs)
        optimizer = train.minimize(train.AdamOptimizer(learning_rate), cost)
    end
    ########## GET it running

    run(sess, initialize_all_variables())

    return sess, tensor_vars
end


const od =(MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.Last())

function train_from_terms!(sess, t::Associative{Symbol}, train_terms_padded, train_hsv; epochs=3)
    costs_o = Float64[]
    @progress "Epochs" for ii in 1:epochs
        @show ii
        data = shuffleobs((train_hsv, train_terms_padded); obsdim=od)
        #data = undersample((train_hsv, train_terms_padded); obsdim=od, shuffleobs=true)
        batchs = eachbatch(data; size=batch_size, obsdim=od)
        @progress "Batches" for (hsv,terms) in batchs
            cost_o, optimizer_o = run(sess,
                [t[:cost], t[:optimizer]],
            Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))
            push!(costs_o, cost_o)
        end
    end
    mean(costs_o)
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
    preds_o = Matrix{Float64}( size(test_terms_padded,1)-1, 0)
    @progress for (hsv,terms) in batchs
        preds, preds_onehot, cost_o = run(sess,
        [t[:Term_preds_s], t[:Term_preds_onehots], t[:cost]],
        Dict(t[:X_hsv]=>hsv, t[:Term_obs_s]=>terms))

        preds_o = [preds_o preds]

        push!(perps, masked_perp(terms, log.(preds_onehot)))
        acc_total += masked_acc(terms, preds)
        cost_total += cost_o
    end

    (cost_total/length(batchs),
    acc_total/length(batchs),
    geomean(perps),
    preds_o)
end

|
