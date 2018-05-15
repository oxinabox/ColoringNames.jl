using TensorFlow
using StatsBase
using Juno
using FileIO
using JLD

const Summaries = TensorFlow.summary

export TermToColorDistributionSOWE

immutable TermToColorDistributionSOWE{OPT, ENC} <: AbstractTermToColorDistributionML
    enc::ENC
    sess::Session
    optimizer::OPT
end


function TermToColorDistributionSOWE(enc, word_vecs=rand(300,nlabel(enc))::AbstractMatrix;
                                     output_res=256,
                                     hidden_layer_size=size(word_vecs,1),
                                     n_steps=-1,
                                     learning_rate=0.01
)
    graph = Graph()
    sess = Session(graph)

    ###################################
    @tf begin
        keep_prob = placeholder(Float32; shape=[])

        terms = placeholder(Int32; shape=[n_steps, -1])
        term_lengths = indmin(terms, 1) - 1 #Last index is the one before the first occurance of 0 (the minimum element) Would be faster if could use find per dimentions

        emb_table = [zeros(size(word_vecs,1))'; word_vecs'] # add an extra-first row for padding
        terms_emb = gather(emb_table, terms+1) # move past the first row we added for zeros)

        H = sum(terms_emb,1)

        W1 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B1 = get_variable((hidden_layer_size), Float32)
        Z1 = nn.dropout(nn.relu(H*W1 + B1), keep_prob)


        W2 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B2 = get_variable((hidden_layer_size), Float32)
        Z2 = nn.dropout(nn.relu(Z1*W2 + B2), keep_prob)
        
        W3 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B3 = get_variable((hidden_layer_size), Float32)
        Z3 = nn.dropout(nn.relu(Z2*W3 + B3), keep_prob)
        
        W4 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B4 = get_variable((hidden_layer_size), Float32)
        Z4 = nn.dropout(nn.relu(Z3*W4 + B4), keep_prob)

        Z=Z4

        function declare_output_layer(name)
            W = get_variable("W_$name", (hidden_layer_size, output_res), Float32)
            B = get_variable("B_$name", (output_res), Float32)
            Y_logit = identity(Z*W + B, name="Yp_logit_$name")
            Y = nn.softmax(Y_logit; name="Yp_$name")
            Yp_obs = placeholder(Float32; shape=[output_res, -1], name="Yp_obs_$name")'
            loss = nn.softmax_cross_entropy_with_logits(;labels=Yp_obs, logits=Y_logit, name="loss_$name")

            Summaries.scalar("loss_$name", reduce_mean(loss); name="summary_loss_$name")
#            Summaries.histogram("W_$name", W; name="summary_W_$name")
            loss
        end
        loss_hue = declare_output_layer("hue")
        loss_sat = declare_output_layer("sat")
        loss_val = declare_output_layer("val")


        cost = reduce_mean(loss_hue + loss_sat + loss_val)
        optimizer = train.minimize(train.AdamOptimizer(learning_rate), cost)

        # Generate some summary operations
        summary_cost = Summaries.scalar("cost", cost)
#        summary_W1 = Summaries.histogram("W1", W1)

    end
    run(sess, global_variables_initializer())
    sess, optimizer


    TermToColorDistributionSOWE(enc, sess, optimizer)
end
