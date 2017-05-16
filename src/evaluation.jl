export descretized_perplexity, find_bin, mse_from_peak, peak, bin_expected_value, total_descretized_logprob

"Determine which bin a continous value belongs in"
function find_bin(value, nbins, range_min=0.0, range_max=1.0)
    #TODO Check boundries
    portion = nbins * value/(range_max-range_min)

    clamp(round(Int, portion), 1, nbins)
end

function bin_expected_value(bin, nbins) #GOLDPLATE consider non 0-1 range_scale
    bin/nbins - 0.5/nbins
end


"""
For obs a Vector of continous variables,
this descretizes each observation into one bin.
For `predicted_class_probs` the predictions of probability for each bin.
Calculated the perplexity.
"""
function descretized_perplexity(obs, predicted_class_probs)
    total_lp = total_descretized_logprob(obs, predicted_class_probs)
    exp2(-total_lp/length(obs))
end

function total_descretized_logprob(obs, predicted_class_probs)
    @assert all(0 .<= obs .<= 1) #GOLDPLATE: deal with non-0-1 ranges
    @assert(length(obs)==size(predicted_class_probs, 1), "$(size(obs)), $(size(predicted_class_probs))")
    output_res = size(predicted_class_probs, 2)
    bin_obs = find_bin(obs, output_res)
    total = 0.0
    for (row, bin) in enumerate(bin_obs)
        total+=log2(predicted_class_probs[row, bin])
    end
    total
end



"""
For `predicted_class_probs` the predictions of probability for each bin.
Finds the bin with the highest probability, and determines the value in continous space for that value
"""
function peak(predicted_class_probs)
    #GOLDPLATE: deal with non-0-1 ranges

    nbins = length(predicted_class_probs)
    bin_expected_value(indmax(predicted_class_probs), nbins)
end

peak(predicted_class_probs::AbstractMatrix) = mapslices(peak, predicted_class_probs, 2)

"Mean squared error"
function mse(obs, preds)
    mean(sumabs2(preds.-obs, 2))
end

function mse_from_peak(obs, predicted_class_probs::AbstractMatrix)
    @assert(length(obs)==size(predicted_class_probs, 1))
    preds = peak(predicted_class_probs)
    mse(obs, preds)
end

function mse_from_peak{T<:AbstractMatrix}(obs::AbstractMatrix, predicted_class_probs::NTuple{3, T})
    @assert size(obs,2)==length(predicted_class_probs)
    preds = reduce(hcat, peak.(predicted_class_probs))
    mse(obs, preds)
end
