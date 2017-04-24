

using Distributions

export gaussianhot, vonmiseshot, gaussianhot!, vonmiseshot!


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

function discretize!(bins::AbstractVector, distr, range_min, range_max)
    nbins = length(bins)
    @assert range_max > range_min
    bin_size = (range_max-range_min)/nbins
    below_cdf = cdf(distr, range_min)
    for ii in 1:nbins
        top_at = range_min + ii*bin_size
        top_cdf = cdf(distr, top_at)
        bin_val = top_cdf - below_cdf
        below_cdf = top_cdf
        bins[ii] = max(0, bin_val) #deal with anything that just slips under due to precision
    end
    return bins
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
