

using Distributions

export gaussianhot, vonmiseshot, gaussianhot!, vonmiseshot!, splay_probabilities


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

"""
Takes in a matrix of HSVs for colors,
and encodes it as if each value was the expected value of a Gaussian (for S and V), or VonVises distribution,
and returns a histogram for each.
"""
function splay_probabilities(hsv, nbins, stddev=1/nbins)
    num_obs =  nobs(hsv, ObsDim.First())
    hp = Matrix{Float32}((nbins, num_obs))
    sp = Matrix{Float32}((nbins, num_obs))
    vp = Matrix{Float32}((nbins, num_obs))
    @progress for (ii, obs) in enumerate(eachobs(hsv, ObsDim.First()))
        vonmiseshot!(@view(hp[:,ii]), hsv[1], stddev)
        gaussianhot!(@view(sp[:,ii]), hsv[2], stddev)
        gaussianhot!(@view(vp[:,ii]), hsv[3], stddev)
    end
    (hp, sp, vp)
end
