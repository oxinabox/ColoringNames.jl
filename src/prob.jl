
import Juno: @progress
using Distributions
using CatViews

export gaussianhot, gaussianhot!,
    vonmiseshot, vonmiseshot!,
    gaussianwraparoundhot, gaussianwraparoundhot!,
    splay_probabilities, splay_probabilities!


function range_scale(val, curlow, curhigh, newlow, newhigh)
    @assert curlow < curhigh
    @assert newlow < newhigh

    @assert curlow <= val <= curhigh
    pos = (val-curlow)/(curhigh-curlow)
    @assert 0 <= pos <= 1
    newval = newlow + pos*(newhigh - newlow)
    newval
end



function discretize(distr, nbins, range_min, range_max)
    discretize!(Vector(nbins), distr, range_min, range_max)
end

function discretize!{T}(bins::AbstractVector{T}, distr, range_min, range_max)
    nbins = length(bins)
    @assert range_max > range_min
    bin_size = (range_max-range_min)/nbins
    below_cdf = cdf(distr, range_min)

    # NB: We zero each element of the bin, then add to it, to allow for aliasing in the bins
    for ii in eachindex(bins)
        @inbounds bins[ii]=zero(T)
    end

    for ii in 1:nbins
        top_at = range_min + ii*bin_size
        top_cdf = cdf(distr, top_at)
        bin_val = top_cdf - below_cdf
        below_cdf = top_cdf
        @inbounds bins[ii] += max(zero(T), bin_val) #deal with anything that just slips under due to precision
    end
    return bins
end


function gaussianwraparoundhot!{T}(bins::AbstractVector{T}, value, range_min=zero(T), range_max=one(T), stddev=(range_max-range_min)/length(bins))
    #Inner range is 3x size of true range
    range_len = range_max - range_min
    inner_min = range_min - range_len
    inner_max = range_max + range_len

    distr = TruncatedNormal(value, stddev, inner_min, inner_max)
    inner_bins = CatView(bins, bins, bins) #have the bins to be fed into be aliased 3 times
    discretize!(inner_bins, distr, inner_min, inner_max)
    bins
end

function gaussianwraparoundhot{T}(value::T, nbins, range_min=zero(T), range_max=one(T), stddev=(range_max-range_min)/nbins)
    gaussianwraparoundhot!(Vector{T}(nbins), value, range_min , range_max, stddev)
end

function gaussianhot{T}(value::T, nbins, range_min=zero(T), range_max=one(T), stddev=(range_max-range_min)/nbins)
    gaussianhot!(Vector{T}(nbins), value, range_min , range_max, stddev)
end

function gaussianhot!{T}(bins::AbstractVector{T}, value, range_min=zero(T), range_max=one(T), stddev=(range_max-range_min)/length(bins))
    distr = TruncatedNormal(value, stddev, range_min, range_max)
    discretize!(bins, distr, range_min, range_max)
end


function vonmiseshot{T}(value::T, nbins, range_min=zero(T), range_max=one(T), stddev=2π/nbins)
    vonmiseshot!(Vector{T}(nbins), value, range_min , range_max, stddev)
end

function vonmiseshot!{T}(bins::AbstractVector{T}, value, range_min=zero(T), range_max=one(T), stddev=2π/length(bins))
    @assert(range_min ≤ value ≤ range_max)
    scaled_value = range_scale(value, range_min, range_max, -π, π)
    @assert(-π ≤ scaled_value ≤ π)
    kappa = inv(stddev)
    distr = VonMises(scaled_value, kappa)
    discretize!(bins, distr, -π, π)
end

"""
Takes in a matrix of HSVs for colors,
and encodes it as if each value was the expected value of a Gaussian (for S and V), or VonVises distribution (for H),
and returns a histogram for each.
"""
function splay_probabilities(hsv, nbins, stddev=1/nbins)
    num_obs =  nobs(hsv, ObsDim.First())
    hp = Matrix{Float32}((nbins, num_obs))
    sp = Matrix{Float32}((nbins, num_obs))
    vp = Matrix{Float32}((nbins, num_obs))

    splay_probabilities!(hp,sp,vp, hsv; stddev=stddev)
end

"""
Inplace verson of `splay_probabilities`
places results into hp, sp, vp for probabilities of hue, saturation and value
"""
function splay_probabilities!(hp,sp,vp, hsv; stddev=1/size(hp,1))
    @assert(size(hp) == size(sp) == size(vp))
    @progress "splaying probabilities" for (ii, obs) in enumerate(eachobs(hsv, ObsDim.First()))
        gaussianwraparoundhot!(@view(hp[:,ii]), hsv[ii, 1], stddev)
        gaussianhot!(@view(sp[:,ii]), hsv[ii, 2], stddev)
        gaussianhot!(@view(vp[:,ii]), hsv[ii, 3], stddev)
    end
    (hp, sp, vp)
end
