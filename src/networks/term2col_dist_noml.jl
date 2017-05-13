using StatsBase
using Juno

const Summaries = TensorFlow.summary

export TermToColorDistributionEmpirical

immutable TermToColorDistributionEmpirical{N}
    output_res::Int
    term2dist::Dict{String, NTuple{N, Vector{Float32}}}
end

function TermToColorDistributionEmpirical(output_res)
    term2dist=Dict{String, NTuple{3, Vector{Float32}}}()
    TermToColorDistributionEmpirical(output_res, term2dist)
end

function train!(mdl::TermToColorDistributionEmpirical, train_terms, train_hsv::AbstractMatrix;
                splay_stddev=1/mdl.output_res)

    train_hsvps = splay_probabilities(train_hsv, mdl.output_res, splay_stddev)
    train!(mdl, train_terms, train_hsvps, epochs=epoch)

end


function train!{N}(mdl::TermToColorDistributionEmpirical{N}, train_terms, train_ps_s::NTuple{N})
    @progress "averaging obs" for (lbl, inds) in labelmap(train_terms)
        !haskey(mdl.term2dist) || error("Distribution for $lbl already trained")
        dists = mdl.term2dist[lbl] = tuple((zeros(Float32, mdl.output_res) for _ in 1:N)...)

        for (dist, train_ps) in zip(dists, train_ps_s)
            for ii in inds
                dist .+= train_ps[ii]
            end
            dist ./= length(inds)
        end
    end
    mdl
end


function query(mdl::TermToColorDistributionEmpirical,  input_text)
    mdl.term2dist[input_text]
end


"Run all evalutations, returning a dictionary of results"
function evaluate{N}(mdl::TermToColorDistributionNetwork{N}, test_terms, test_hsv)
    Yps = [Matrix{Float32}(0, mdl.output_res) for ii in 1:N]
    
    for term in test_terms
        for (ii, ps) in enumerate(query(mdl, term))
            Yps[ii] = [Yps[ii]; ps]
        end
    end

    Yp_uniform = ones(Yps[1])./length(test_terms)

    perps = [descretized_perplexity(Y_obs, Yp) for (Y_obs,Yp) in  obsview(test_hsv, Yps)]
    @names_from begin
        perp_hue = perps[1]
        perp_sat = perps[2]
        perp_val = perps[3]
        perp = geomean(perps)

        perp_uniform_baseline = descretized_perplexity(test_hsv[1,:], Yp_uniform)

        mse_to_peak = mse_from_peak(test_hsv, tuple(Yps...))
    end
end