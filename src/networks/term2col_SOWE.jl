using TensorFlow
using StatsBase
using Juno
using FileIO
using JLD

const Summaries = TensorFlow.summary

export TermToColorDistributionSOWE

struct TermToColorDistributionSOWE{OPT, ENC, SMR} <: AbstractDistEstML
    encoding::ENC
    sess::Session
    optimizer::OPT
    summary::SMR
end



SOWE_combine_terms(hidden_layer_size) = function (terms_emb, keep_prob)
    @tf begin
        H = sum(terms_emb,1)
        
        function layer(H, name)
            W = get_variable("W_$name", (hidden_layer_size, hidden_layer_size), Float32)
            B = get_variable("b_$name", (hidden_layer_size), Float32)
            Z = nn.dropout(nn.relu(H*W + B), keep_prob; name="Z_$name")
        end
        
        Z = H
        for ii in 1:1
            Z = layer(Z,ii)
        end
        
        Z
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

#########################
struct TermToColorPointSOWE{OPT, ENC, SMR} <: AbstractPointEstML
    encoding::ENC
    sess::Session
    optimizer::OPT
    summary::SMR
end

function TermToColorPointSOWE(enc, word_vecs=rand(300,nlabel(enc))::AbstractMatrix;
                                     hidden_layer_size=size(word_vecs,1),
                                     n_steps=-1,
                                     learning_rate=0.001
)
    sess, optimizer, summary_op = init_point_est_network(
        SOWE_combine_terms(hidden_layer_size),
        word_vecs, n_steps, hidden_layer_size, learning_rate
        )
   
    TermToColorPointSOWE(enc, sess, optimizer, summary_op)
end

