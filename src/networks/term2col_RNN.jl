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
    function layer(H, name)
        W = get_variable("W_$name", (hidden_layer_size, hidden_layer_size), Float32)
        B = get_variable("b_$name", (hidden_layer_size), Float32)
        Z = nn.dropout(nn.relu(H*W + B), keep_prob; name="Z_$name")
    end
    
    @tf begin
        # 0 is the min term, indmin finds its index
        # We include that index in the terms, as it serves as a demarcation pseudotoken
        # so the network knows where it ends
        # This marginally improves results
        term_lengths = indmin(terms_emb, 1)[:, 1]
        
#        map(unstack(term_embs)) do 
#        stack(
        
        X= term_embs
        cell = nn.rnn_cell.DropoutWrapper(nn.rnn_cell.GRUCell(hidden_layer_size), keep_prob)
        
        Hs, state = nn.rnn(cell, X, term_lengths; dtype=Float32, time_major=true)
        Ho = Hs[end]      
        H = reshape(Ho, [-1, hidden_layer_size]) #Force shape
        
        W1 = get_variable((hidden_layer_size, hidden_layer_size), Float32)
        B1 = get_variable((hidden_layer_size), Float32)
        Z1 = nn.dropout(nn.relu(H*W1 + B1), keep_prob)
        
        Z1
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


##############################################

immutable TermToColorPointRNN{OPT, ENC, SMR} <: AbstractPointEstML
    encoding::ENC
    sess::Session
    optimizer::OPT
    summary::SMR
end

function TermToColorPointRNN(enc, word_vecs=rand(300,nlabel(enc))::AbstractMatrix;
                                     hidden_layer_size=size(word_vecs,1),
                                     n_steps=-1,
                                     learning_rate=0.001
)
    sess, optimizer, summary_op = init_point_est_network(
        RNN_combine_terms(hidden_layer_size),
        word_vecs, n_steps, hidden_layer_size, learning_rate
        )
   
    TermToColorPointRNN(enc, sess, optimizer, summary_op)
end

