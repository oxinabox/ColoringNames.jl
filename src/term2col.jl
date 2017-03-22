
function terms_to_color_network(n_classes, n_steps;
        batch_size = 128,
        hidden_layer_size = 256,
        embedding_dim = 16,
        learning_rate=0.05
    )


    ###################################
    sess = Session(Graph())
    @tf begin
        emb_table = get_variable((n_classes, embedding_dim), Float32)
        terms = placeholder(Int32, shape=(batch_size, n_steps))
        term_lengths = indmin(,2) - 1 #Last index is the one before the first occurance of 0 (the minimum element) Would be faster if could use find per dimentions


        terms_emb = gather(emb_table, terms+1)

        cell = nn.rnn_cell.GRUCell(hidden_layer_size)
        Hs, states = nn.rnn(cell, terms_emb, term_lengths; dtype=Float32)
        W = get_variable((hidden_layer_size,2+1+1), Float32)
        Ws = sigmoid(W*Hs[end])

        Math.atan2(Ws[1,:])

    end
    run(sess, global_variables_initializer())
    sess
end
