using TensorFlow
using StatsBase
using Juno
using FileIO
using JLD

const Summaries = TensorFlow.summary

export TermToColorDistributionSOWE

immutable TermToColorDistributionRNN{OPT, ENC, SMR} <: AbstractDistEstML
    encoding::ENC
    sess::Session
    optimizer::OPT
    summary::SMR
end

RNN_combine_terms(hidden_layer_size) = function (terms_emb, keep_prob)
    @tf begin
        term_lengths = indmin(terms_emb, 1)[:, 1] - 1 #Last index is the one before the first occurance of 0 (the minimum element)
        
        cell = nn.rnn_cell.DropoutWrapper(nn.rnn_cell.GRUCell(hidden_layer_size), keep_prob)
        Hs, state = nn.rnn(cell, terms_emb, term_lengths; dtype=Float32, time_major=true)
        H = Hs[end]

        W1 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B1 = get_variable((hidden_layer_size), Float32)
        Z1 = nn.dropout(nn.relu(H*W1 + B1), keep_prob)


        W2 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B2 = get_variable((hidden_layer_size), Float32)
        Z2 = nn.dropout(nn.relu(Z1*W2 + B2), keep_prob)

        W3 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B3 = get_variable((hidden_layer_size), Float32)
        Z3 = nn.dropout(nn.relu(Z2*W3 + B3), keep_prob)

        Z3
    end
end

function TermToColorDistributionRNN(enc, word_vecs=rand(300,nlabel(enc))::AbstractMatrix;
                                     output_res=256,
                                     hidden_layer_size=size(word_vecs,1),
                                     n_steps=-1,
                                     learning_rate=0.001
)
    
    sess, optimizer, summary_op = init_dist_est_network(
        RNN_combine_terms(hidden_layer_size),
        word_vecs, n_steps, hidden_layer_size, output_res, learning_rate
        )
   
    TermToColorDistributionRNN(enc, sess, optimizer, summary_op)
end
