using TensorFlow
using StatsBase
using Juno
using FileIO
using JLD

const Summaries = TensorFlow.summary

export TermToColorDistributionSOWE

immutable TermToColorDistributionSOWE{OPT, ENC, SMR} <: AbstractDistEstML
    encoding::ENC
    sess::Session
    optimizer::OPT
    summary::SMR
end

SOWE_combine_terms(hidden_layer_size) = function (terms_emb, keep_prob)
    @tf begin
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
    end
end

function TermToColorDistributionSOWE(enc, word_vecs=rand(300,nlabel(enc))::AbstractMatrix;
                                     output_res=256,
                                     hidden_layer_size=size(word_vecs,1),
                                     n_steps=-1,
                                     learning_rate=0.001
)
    sess, optimizer, summary_op = init_dist_est_network(
        SOWE_combine_terms(hidden_layer_size),
        word_vecs, n_steps, hidden_layer_size, output_res, learning_rate
        )
   
    TermToColorDistributionSOWE(enc, sess, optimizer, summary_op)
end
