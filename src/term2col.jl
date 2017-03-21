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

        terms_emb = gather(emb_table, terms+1)

        cell = nn.rnn_cell.LSTMCell(hidden_layer_size)
        nn.dynamic_rnn(cell, terms_emb)
        

    end
    run(sess, global_variables_initializer())
    sess
end
