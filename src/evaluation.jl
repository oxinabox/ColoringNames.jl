"Determine which bin a continous value belongs in"
function find_bin(data::Vector, nbins)
    midpoints = KernelDensity.kde_range((0,1), nbins)
    map(data) do x
        k = searchsortedfirst(midpoints,x)
        if k==1
            return 1
        end
        below_mp = midpoints[k-1]
        above_mp = midpoints[k]
        @assert below_mp ≤ x ≤ above_mp
        if abs(above_mp-x) < abs(x-below_mp)
            k #closer to above
        else
            k-1 # closer to below
        end
    end
end

function bin_expected_value(bin, nbins)
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
        pred_prob = predicted_class_probs[row, bin]
        if pred_prob==0
            @show row, bin
            @show pred_prob
        end
        total+=log2(pred_prob)
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
    mean(sum(abs2.(preds.-obs), 2))
end

function mse_from_peak{T<:AbstractMatrix}(obs::AbstractMatrix, predicted_class_probs::NTuple{3, T})
    preds = reduce(hcat, peak.(predicted_class_probs))
    @assert size(preds, 1) == size(obs,1) "$(size(preds,1)) != $(size(obs,1))"
    mse(obs, preds)
end
