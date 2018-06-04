"Determine which bin a continous value belongs in"
function find_bin(data::AbstractVector, nbins)
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
    midpoints = KernelDensity.kde_range((0,1), nbins)
    midpoints[bin]
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


function full3d_descretized_perplexity(obs, predicted_class_probs)
    Y_obs_hue, Y_obs_sat, Y_obs_val = obs
    Yp_hue, Yp_sat, Yp_val = predicted_class_probs
    num_obs = length(Y_obs_hue)
    @assert num_obs == length(Y_obs_sat) == length(Y_obs_val)
    
    total_lp = 0.0
    total_lp += total_descretized_logprob(Y_obs_hue, Yp_hue)
    total_lp += total_descretized_logprob(Y_obs_sat, Yp_sat)
    total_lp += total_descretized_logprob(Y_obs_val, Yp_val)
    
    exp2(-total_lp/num_obs)
end

###########################################
# Peak /Mode

"""
The distribiutions mode
For `predicted_class_probs` the predictions of probability for each bin.
Finds the bin with the highest probability, and determines the value in continous space for that value
"""
function peak(predicted_class_probs)
    #GOLDPLATE: deal with non-0-1 ranges

    nbins = length(predicted_class_probs)
    bin_expected_value(indmax(predicted_class_probs), nbins)
end

peak(predicted_class_probs::AbstractMatrix) = mapslices(peak, predicted_class_probs, 2)




function mse_from_peak{T<:AbstractMatrix}(obs::AbstractMatrix, predicted_class_probs::NTuple{3, T})
    preds = reduce(hcat, peak.(predicted_class_probs))
    @assert size(preds, 1) == size(obs,1) "$(size(preds,1)) != $(size(obs,1))"
    mse(obs, preds)
end

#######################################################
# Weight Mean

function weighted_bin_mean(bin_weights::AbstractVector)
    @assert(all(0 .<= bin_weights .<= 1))
    nbins = length(bin_weights)
    midpoints = KernelDensity.kde_range((0,1), nbins)
    (midpoints ⋅ bin_weights)
end

function weighted_bin_mean_hue(bin_weights::AbstractVector)
    @assert(all(0 .<= bin_weights .<= 1))
    nbins = length(bin_weights)
    midpoints = KernelDensity.kde_range((0,1), nbins)
    
    c_midpoints = cos.(2π.*midpoints)
    c_mean =  (c_midpoints ⋅ bin_weights)

    s_midpoints = sin.(2π.*midpoints)
    s_mean =  (s_midpoints ⋅ bin_weights)
    
    mod(atan2(s_mean, c_mean), 2π)/2π
end

function distmean(hp::AbstractMatrix, sp::AbstractMatrix, vp::AbstractMatrix)
    hcat(
        mapslices(weighted_bin_mean_hue, hp, 2),
        mapslices(weighted_bin_mean, sp, 2),
        mapslices(weighted_bin_mean, vp, 2)
    )
end
function mse_from_distmean{T<:AbstractMatrix}(obs::AbstractMatrix, predicted_class_probs::NTuple{3, T})
    preds = distmean(predicted_class_probs...)
    @assert size(preds, 1) == size(obs,1) "$(size(preds,1)) != $(size(obs,1))"
    mse(obs, preds)
end

#############

"Mean squared error"
function mse(obs, pred)
    mean(hsv_squared_error(pred, obs))
end

hsquared_error(ha::AbstractVector, hb::AbstractVector) = @. min((ha - hb)^2, (ha - hb - 1)^2)
hsquared_error(ha, hb) = min((ha - hb)^2, (ha - hb - 1)^2)

squared_error(a::AbstractVector, b::AbstractVector) = @. (a-b)^2
squared_error(a, b) = (a-b)^2


function hsv_squared_error(aa, bb)
    ha = aa[:, 1]
    sa = aa[:, 2]
    va = aa[:, 3]

    hb = bb[:, 1]
    sb = bb[:, 2]
    vb = bb[:, 3]
    
    hsquared_error(ha, hb) + squared_error(sa, sb) + squared_error(va, vb)
end


##################

