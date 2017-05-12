using TensorFlow
using StatsBase
using Juno
using FileIO
using JLD

const Summaries = TensorFlow.summary

export TermToColorDistributionNetwork, train_to_color_dist!, querier, evaluate

immutable TermToColorDistributionNetwork{NTerms, S<:AbstractString, OPT}
    encoding::LabelEnc.NativeLabels{S, NTerms}
    sess::Session
    optimizer::OPT
    max_tokens::Int #Max nummber of tokens in a description
    output_res::Int
    hidden_layer_size::Int
    embedding_dim::Int
    batch_size::Int
    learning_rate::Float32 #TODO: Work out a way for this not to be a parameter of the network, but of training
end

function FileIO.save(mdl::TermToColorDistributionNetwork, save_dir; extra_info...)
    params = Dict(string(nn)=>getfield(mdl,nn) for nn in fieldnames(mdl) if !(nn in ["sess", "optimizer"]))
    for (kk, vv) in extra_info
        params[string(kk)] = vv
    end
    params["save_time"] = now()
    params["git_hash"] = strip(readstring(`git rev-parse --verify HEAD`))
    
    params["model_path"] = joinpath(save_dir, "model.jld")
    save(joinpath(save_dir, "params.jld", params))
    
    train.save(train.Saver(), sess, params["model_path"])
end


function restore(::Type{TermToColorDistributionNetwork}, param_path, model_path=load(param_path,"model_path"))
    @load(param_path, encoding, max_tokens, batch_size, hidden_layer_size, embedding_dim, output_res, learning_rate)
    
    sess, optimizer = init_terms_to_color_dist_network_session(nlabel(encoding), max_tokens, batch_size, hidden_layer_size, embedding_dim, output_res, learning_rate)
    train.restore(train.Saver(), sess, model_path)

    TermToColorDistributionNetwork(encoding, sess, optimizer,  max_tokens, output_res, hidden_layer_size, embedding_dim, batch_size, learning_rate)
end


function TermToColorDistributionNetwork{S<:AbstractString, NTerms}(encoding::LabelEnc.NativeLabels{S, NTerms};
                                                max_tokens=4,
                                                output_res=64,
                                                hidden_layer_size=128, #* at from search parameter space on dev set at output_res 64
                                                embedding_dim=16, #* ditto
                                                batch_size=12_381,
                                                learning_rate=0.5)

    sess, optimizer = init_terms_to_color_dist_network_session(NTerms, max_tokens, batch_size, hidden_layer_size, embedding_dim, output_res, learning_rate)
    TermToColorDistributionNetwork(encoding, sess, optimizer,  max_tokens, output_res, hidden_layer_size, embedding_dim, batch_size, learning_rate)
end

function init_terms_to_color_dist_network_session(
        n_term_classes,
        n_steps,
        batch_size,
        hidden_layer_size = 512,
        embedding_dim = 16,
        output_res = 256,
        learning_rate=0.05,
    )

    graph = Graph()
    sess = Session(graph)

    ###################################

    @tf begin
        keep_prob = placeholder(Float32; shape=[])

        terms = placeholder(Int32; shape=[n_steps, batch_size])
        term_lengths = indmin(terms, 1) - 1 #Last index is the one before the first occurance of 0 (the minimum element) Would be faster if could use find per dimentions

        emb_table = get_variable((n_term_classes, embedding_dim), Float32)
        terms_emb = gather(emb_table, terms+1)

        cell = nn.rnn_cell.DropoutWrapper(nn.rnn_cell.GRUCell(hidden_layer_size), keep_prob)
        Hs, states = nn.rnn(cell, terms_emb, term_lengths; dtype=Float32, time_major=true)

        W1 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B1 = get_variable((hidden_layer_size), Float32)
        Z1 = nn.dropout(nn.relu(Hs[end]*W1 + B1), keep_prob)


        function declare_output_layer(name)
            W = get_variable("W_$name", (hidden_layer_size, output_res), Float32)
            B = get_variable("B_$name", (output_res), Float32)
            Y_logit = Z1*W + B
            Y = nn.softmax(Y_logit; name="Yp_$name")
            Yp_obs = placeholder(Float32; shape=[output_res, batch_size], name="Yp_obs_$name")'
            loss = nn.softmax_cross_entropy_with_logits(;labels=Yp_obs, logits=Y_logit, name="loss_$name")

            Summaries.scalar("loss_$name", reduce_mean(loss); name="summary_loss_$name")
#            Summaries.histogram("W_$name", W; name="summary_W_$name")
            loss
        end
        loss_hue = declare_output_layer("hue")
        loss_sat = declare_output_layer("sat")
        loss_val = declare_output_layer("val")


        cost = reduce_mean(loss_hue + loss_sat + loss_val)
        optimizer = train.minimize(train.AdamOptimizer(), cost)

        # Generate some summary operations
        summary_cost = Summaries.scalar("cost", cost)
#        summary_W1 = Summaries.histogram("W1", W1)

    end
    run(sess, global_variables_initializer())
    sess, optimizer
end



function train!(mdl::TermToColorDistributionNetwork, train_terms_padded, train_hsv::AbstractMatrix,
                            log_dir=nothing;
                            epochs=3, dropout_keep_prob=0.5f0, splay_stddev=1/mdl.output_res)

    train_hsvps = splay_probabilities(train_hsv, mdl.output_res, splay_stddev)
    train_to_color_dist!(mdl, train_terms_padded, train_hsvps, log_dir;
                        epochs=epochs, dropout_keep_prob=dropout_keep_prob)

end


function train!(mdl::TermToColorDistributionNetwork, train_terms_padded, train_hsvps::NTuple{3},
                                log_dir=nothing;
                                epochs=30, #From checking convergance at default parameters for network
                                dropout_keep_prob=0.5f0)
    ss = mdl.sess.graph
    if log_dir!=nothing
        summary_op = Summaries.merge_all() #XXX: Does this break if the default graph has changed?
        summary_writer = Summaries.FileWriter(log_dir; graph=ss)
    else
        warn("No log_dir set during training; no logs will be kept.")
    end

    costs_o = Float64[]

    @progress "Epochs" for epoch_ii in 1:epochs

        data = shuffleobs((train_hsvps..., train_terms_padded))
        batchs = eachbatch(data; size=mdl.batch_size)

        @progress "Batches" for (hp_obs, sp_obs, vp_obs, terms) in batchs

            cost_o, optimizer_o = run(
                mdl.sess,
                [
                    ss["cost"],
                    mdl.optimizer
                ],
                Dict(
                    ss["keep_prob"]=>dropout_keep_prob,
                    ss["terms"]=>terms,
                    ss["Yp_obs_hue"]=>hp_obs,
                    ss["Yp_obs_sat"]=>sp_obs,
                    ss["Yp_obs_val"]=>vp_obs
                )
            )

            push!(costs_o, cost_o)
        end

        #Log summary
        if log_dir!=nothing
            (hp_obs, sp_obs, vp_obs, terms) = first(batchs) #use the first batch to eval on, the one we trained least recently.  they are shuffled every epoch anyway
            summaries = run(mdl.sess, summary_op,
                    Dict(
                        ss["keep_prob"]=>1.0,
                        ss["terms"]=>terms,
                        ss["Yp_obs_hue"]=>hp_obs,
                        ss["Yp_obs_sat"]=>sp_obs,
                        ss["Yp_obs_val"]=>vp_obs
                    )
                )

            write(summary_writer, summaries, epoch_ii)
        end
    end
    costs_o
end


function query(mdl::TermToColorDistributionNetwork,  input_text)
    label = input_text
    labels, _ = prepare_labels([label], mdl.encoding, do_demacate=false)

    nsteps_to_pad = mdl.max_tokens - size(labels,1)
    nbatch_items_to_pad = mdl.batch_size - size(labels,2)

    padded_labels = [[labels; zeros(Int, nsteps_to_pad)] zeros(Int, (mdl.max_tokens, nbatch_items_to_pad))]
    ss=mdl.sess.graph
    hp, sp, vp = run(
        mdl.sess,
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


"Run all evalutations, returning a dictionary of results"
function evaluate(mdl::TermToColorDistributionNetwork, test_terms_padded, test_hsv)
    #GOLDPLATE: do this without just storing up results, particularly without doing it via row appends
    
    gg=sess.graph

    Y_obs_hue = Vector{Float32}(); sizehint!(Y_obs_hue, size(test_hsv, 1))
    Y_obs_sat = Vector{Float32}(); sizehint!(Y_obs_sat, size(test_hsv, 1))
    Y_obs_val = Vector{Float32}(); sizehint!(Y_obs_val, size(test_hsv, 1))
    Yp_hue = Matrix{Float32}(0, mdl.output_res)
    Yp_sat = Matrix{Float32}(0, mdl.output_res)
    Yp_val = Matrix{Float32}(0, mdl.output_res)
    

    data = shuffleobs((test_hsv[:, 1], test_hsv[:,2], test_hsv[:,3], test_terms_padded))
    # Shuffle because during validation items that don't fit in batches will be dropped
    # Don't do this at test time; but it is fine when using the validation data to estimate
    batchs = eachbatch(data; size=mdl.batch_size)

    @progress "Batches" for (Y_obs_hue_b, Y_obs_sat_b, Y_obs_val_b, terms) in batchs
        Yp_hue_b, Yp_sat_b, Yp_val_b = run(sess, [gg["Yp_hue"], gg["Yp_sat"], gg["Yp_val"]],  Dict(gg["terms"]=>terms, gg["keep_prob"]=>1.0))

        append!(Y_obs_hue, Y_obs_hue_b)
        append!(Y_obs_sat, Y_obs_sat_b)
        append!(Y_obs_val, Y_obs_val_b)
        Yp_hue = [Yp_hue; Yp_hue_b]
        Yp_sat = [Yp_sat; Yp_sat_b]
        Yp_val = [Yp_val; Yp_val_b]
    end

    Yp_uniform = ones(Yp_hue)./length(Y_obs_hue)

    @names_from begin
        perp_hue = descretized_perplexity(Y_obs_hue, Yp_hue)
        perp_sat = descretized_perplexity(Y_obs_sat, Yp_sat)
        perp_val = descretized_perplexity(Y_obs_val, Yp_val)
        perp = geomean([perp_hue perp_sat perp_val])

        perp_uniform_baseline = descretized_perplexity(Y_obs_hue, Yp_uniform)

        mse_to_peak = mse_from_peak([Y_obs_hue Y_obs_sat Y_obs_val], (Yp_hue, Yp_sat, Yp_val))
    end
end
