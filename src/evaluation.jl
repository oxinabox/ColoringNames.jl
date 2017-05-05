export descretized_perplexity

"""
For obs a Vector of continous variables,
this descretizes each observation into one bin.
For `predicted_class_probs` the predictions of probability for each bin.
Calculated the perplexity.
"""
function descretized_perplexity(obs, predicted_class_probs)
    @assert(length(obs)==size(predicted_class_probs, 1))
    output_res = size(predicted_class_probs, 2)
    bin_obs = find_bin(obs, output_res)
    total = 0.0
    for (row, bin) in enumerate(bin_obs)
        total+=log2(predicted_class_probs[row, bin])
    end
    exp2(-total/length(obs))
end
