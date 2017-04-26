
function terms_to_color_dist_network(n_classes, n_steps;
        batch_size = 128,
        hidden_layer_size = 256,
        embedding_dim = 16,
        output_res = 256,
        learning_rate=0.05
    )


    ###################################
    sess = Session(Graph())
    @tf begin

        terms = placeholder(Int32; shape=[n_steps, batch_size])
        term_lengths = indmin(terms, 1) - 1 #Last index is the one before the first occurance of 0 (the minimum element) Would be faster if could use find per dimentions

        emb_table = get_variable((n_classes, embedding_dim), Float32)
        terms_emb = gather(emb_table, terms+1)

        cell = nn.rnn_cell.GRUCell(hidden_layer_size)
        Hs, states = nn.rnn(cell, terms_emb, term_lengths; dtype=Float32, time_major=true)

        Z = Hs[end]

        Whue = get_variable((hidden_layer_size, output_res), Float32)
        Bhue = get_variable((output_res), Float32)
        Wsat = get_variable((hidden_layer_size, output_res), Float32)
        Bsat = get_variable((output_res), Float32)
        Wval = get_variable((hidden_layer_size, output_res), Float32)
        Bval = get_variable((output_res), Float32)

        Yhue_logit = Z*Whue + Bhue
        Ysat_logit = Z*Wsat + Bsat
        Yval_logit = Z*Wval + Bval

        Yhue = nn.softmax(Yhue_logit)
        Ysat = nn.softmax(Ysat_logit)
        Yval = nn.softmax(Yval_logit)
        

        Yhue = Base.Math.atan2(Z[:,2], Z[:,1])/π + 1 #bring to 0,1 range
        Ysatval = nn.sigmoid(Z[:,3:4])
        Yhuesatval = concat([expand_dims(Yhue, 2), Ysatval], 2)

        Yhuesatval_obs = placeholder(Float32; shape=[batch_size, 3])

        cost = reduce_mean(squared_difference(Yhuesatval_obs, Yhuesatval))
        optimizer = train.minimize(train.AdamOptimizer(), cost)
    end
    run(sess, global_variables_initializer())
    sess, optimizer
end



function train_from_cols!(sess, train_terms_padded, train_hsv; epochs=3)
    ss = sess.graph
    costs_o = Float64[]
    @progress "Epochs" for ii in 1:epochs
        @show ii
        data = shuffleobs((train_hsv, train_terms_padded); obsdim=od)
        #data = undersample((train_hsv, train_terms_padded); obsdim=od, shuffleobs=true)
        batchs = eachbatch(data; size=batch_size, obsdim=od)
        @progress "Batches" for (hsv,terms) in batchs
            cost_o, optimizer_o = run(sess,
                [ss["cost"], ss["optimizer"]],
            Dict(ss["Yhuesatval_obs"]=>hsv, ss["terms"]=>terms))
            push!(costs_o, cost_o)
        end
    end
    mean(costs_o)
end
