using TensorFlow
using StatsBase
using Juno

const Summaries = TensorFlow.summary

export terms_to_color_dist_network, train_to_color_dist!, querier, evaluate


function terms_to_color_dist_network(n_term_classes, n_steps;
        batch_size = 128,
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



function train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsv::AbstractMatrix,
                            log_dir=nothing;
                            epochs=3, dropout_keep_prob=0.5f0, splay_stddev=1/output_res)

    train_hsvps = splay_probabilities(train_hsv, output_res, splay_stddev)
    train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsvps, log_dir;
                        epochs=epochs, dropout_keep_prob=dropout_keep_prob)

end



# Create a summary writer




function train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsvps::NTuple{3},
                                log_dir=nothing;
                                epochs=3, dropout_keep_prob=0.5f0)
    ss = sess.graph
    if log_dir!=nothing
        summary_op = Summaries.merge_all() #XXX: Does this break if the default graph has changed?
        summary_writer = train.SummaryWriter(log_dir)
    else
        warn("No log_dir set during training; no logs will be kept.")
    end

    costs_o = Float64[]

    @progress "Epochs" for epoch_ii in 1:epochs

        data = shuffleobs((train_hsvps..., train_terms_padded))
        batchs = eachbatch(data; size=batch_size)

        @progress "Batches" for (hp_obs, sp_obs, vp_obs, terms) in batchs

            cost_o, optimizer_o = run(
                sess,
                [
                    ss["cost"],
                    optimizer
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
            summaries = run(sess, summary_op,
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


"Run all evalutations, returning a dictionary of results"
function evaluate(sess, test_terms_padded, test_hsv)
    Y_obs_hue = test_hsv[:, 1]
    Y_obs_sat = test_hsv[:, 2]
    Y_obs_val = test_hsv[:, 3]

    gg=sess.graph
    Yp_hue, Yp_sat, Yp_val = run(sess, [gg["Yp_hue"], gg["Yp_sat"], gg["Yp_val"]],  Dict(gg["terms"]=>test_terms_padded, gg["keep_prob"]=>1.0))
    Yp_uniform = ones(Yp_hue)./length(Y_obs_hue)

    @names_from begin
        perp_hue = descretized_perplexity(Y_obs_hue, Yp_hue)
        perp_sat = descretized_perplexity(Y_obs_sat, Yp_sat)
        perp_val = descretized_perplexity(Y_obs_val, Yp_val)
        perp = geomean([perp_hue perp_sat perp_val])

        perp_uniform_baseline = descretized_perplexity(Y_obs_hue, Yp_uniform)

        mse_to_peak = mse_from_peak(test_hsv, (Yp_hue, Yp_sat, Yp_val))
    end
end
