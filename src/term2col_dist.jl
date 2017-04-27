
function terms_to_color_dist_network(n_term_classes, n_steps;
        batch_size = 128,
        hidden_layer_size = 512,
        embedding_dim = 16,
        output_res = 256,
        learning_rate=0.05
    )

    graph = Graph()
    sess = Session(graph)

    ###################################

    @tf begin

        terms = placeholder(Int32; shape=[n_steps, batch_size])
        term_lengths = indmin(terms, 1) - 1 #Last index is the one before the first occurance of 0 (the minimum element) Would be faster if could use find per dimentions

        emb_table = get_variable((n_term_classes, embedding_dim), Float32)
        terms_emb = gather(emb_table, terms+1)

        cell = nn.rnn_cell.GRUCell(hidden_layer_size)
        Hs, states = nn.rnn(cell, terms_emb, term_lengths; dtype=Float32, time_major=true)

        Z = Hs[end]
        function declare_output_layer(name)
            W = get_variable("W_$name", (hidden_layer_size, output_res), Float32)
            B = get_variable("B_$name", (output_res), Float32)
            Y_logit = Z*W + B
            Y = nn.softmax(Y_logit; name="Yp_$name")
            Y_obs = placeholder(Float32; shape=[output_res, batch_size], name="Yp_obs_$name")'
            loss = nn.softmax_cross_entropy_with_logits(;labels=Y_obs, logits=Y_logit, name="loss_$name")
        end
        loss_hue = declare_output_layer("hue")
        loss_sat = declare_output_layer("sat")
        loss_val = declare_output_layer("val")


        cost = reduce_mean(loss_hue + loss_sat + loss_val)
        optimizer = train.minimize(train.AdamOptimizer(), cost)
    end
    run(sess, global_variables_initializer())
    sess, optimizer
end



function train_to_color_dist!(sess, optimizer, batch_size, output_res, train_terms_padded, train_hsv; epochs=3)
    ss = sess.graph
    costs_o = Float64[]

    hp_obs = Matrix{Float32}((output_res, batch_size))
    sp_obs = Matrix{Float32}((output_res, batch_size))
    vp_obs = Matrix{Float32}((output_res, batch_size))


    @progress "Epochs" for ii in 1:epochs
        @show ii
        data = shuffleobs((train_hsv, train_terms_padded); obsdim=od)
        batchs = eachbatch(data; size=batch_size, obsdim=od)

        @progress "Batches" for (hsv, terms) in batchs
            splay_probabilities!(hp_obs, sp_obs, vp_obs, hsv)

            cost_o, optimizer_o = run(
                sess,
                [
                    ss["cost"],
                    optimizer
                ],
                Dict(
                    ss["terms"]=>terms,
                    ss["Yp_obs_hue"]=>hp_obs,
                    ss["Yp_obs_sat"]=>sp_obs,
                    ss["Yp_obs_val"]=>vp_obs
                )
            )

            push!(costs_o, cost_o)
        end
    end
    costs_o
end
