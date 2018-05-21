using TensorFlow
using StatsBase
using Juno
using FileIO
using JLD

const Summaries = TensorFlow.summary

export TermToColorDistributionSOWE

immutable TermToColorDistributionCNN{OPT, ENC, SMR} <: AbstractDistEstML
    encoding::ENC
    sess::Session
    optimizer::OPT
    summary::SMR
end



CNN_combine_terms(hidden_layer_size, n_steps) = function (terms_emb, keep_prob)

    num_filters = 64
    filter_sizes=1:n_steps
    num_filters_total = num_filters * length(filter_sizes)
    
    @tf begin
        X = permutedims(expand_dims(terms_emb,4), [2,1,3,4])
        ##################
        pooled_outputs = map(enumerate(filter_sizes)) do arg
            (i, filter_size) = arg
            # Convolution Layer
            filter_shape = [filter_size, hidden_layer_size, 1, num_filters]
            W = get_variable("W_$i", filter_shape, Float32)
            b = get_variable("b_$i", num_filters, Float32)

            conv = nn.conv2d(
                X,
                W;
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv_$i")
            # Apply nonlinearity
            h = nn.relu(conv + b, name="h_$i")
            # Max-pooling over the outputs
            nn.max_pool(
                h,
                ksize=[1, n_steps - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool_$i")
        end

        # Combine all the pooled features
        h_pool = cat(Tensor, 3, pooled_outputs...)
        h_pool_flat = reshape(h_pool, [-1, num_filters_total])
        ############

        Wo = get_variable((num_filters_total, hidden_layer_size), Float32)
        bo = get_variable(hidden_layer_size, Float32)

        Zo = nn.relu(h_pool_flat*Wo + bo)
    end

end

function TermToColorDistributionCNN(enc, word_vecs=rand(300,nlabel(enc))::AbstractMatrix;
                                     output_res=256,
                                     hidden_layer_size=size(word_vecs,1),
                                     n_steps=4,
                                     learning_rate=0.001
)
    sess, optimizer, summary_op = init_dist_est_network(
        CNN_combine_terms(hidden_layer_size, n_steps),
        word_vecs, n_steps, hidden_layer_size, output_res, learning_rate
        )
   
    TermToColorDistributionCNN(enc, sess, optimizer, summary_op)
end
