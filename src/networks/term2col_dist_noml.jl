using StatsBase
using Juno

const Summaries = TensorFlow.summary

export TermToColorDistributionEmpirical, laplace_smooth

immutable TermToColorDistributionEmpirical{N}
    output_res::Int
    term2dist::Dict{String, NTuple{N, Vector{Float32}}}
end

function TermToColorDistributionEmpirical(output_res=64)
    term2dist=Dict{String, NTuple{3, Vector{Float32}}}()
    TermToColorDistributionEmpirical(output_res, term2dist)
end

function train!(mdl::TermToColorDistributionEmpirical, train_terms, train_hsv::AbstractMatrix;
                splay_stddev=1/mdl.output_res)

    train_hsvps = splay_probabilities(train_hsv, mdl.output_res, splay_stddev)
    train!(mdl, train_terms, train_hsvps)

end


function train!{N}(mdl::TermToColorDistributionEmpirical{N}, train_terms, train_ps_s::NTuple{N})
    @progress "averaging obs" for (lbl, inds) in labelmap(train_terms)
        !haskey(mdl.term2dist, lbl) || error("Distribution for $lbl already trained")
        dists = mdl.term2dist[lbl] = tuple((zeros(Float32, mdl.output_res) for _ in 1:N)...)

        for (dist, train_ps) in zip(dists, train_ps_s)
            for ii in inds
                dist .+= train_ps[:, ii]
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
function evaluate{N}(mdl::TermToColorDistributionEmpirical{N}, test_terms, test_hsv)
    Yps = [Matrix{Float32}(length(test_terms), mdl.output_res) for ii in 1:N]
    
    for (term_ii, term) in enumerate(test_terms)
        for (dist_ii, ps) in enumerate(query(mdl, term))
            Yps[dist_ii][term_ii, :] = ps'
        end
    end

    @names_from begin
        perp_hue = descretized_perplexity(test_hsv[:,1], Yps[1])
        perp_sat = descretized_perplexity(test_hsv[:,2], Yps[2])
        perp_val = descretized_perplexity(test_hsv[:,3], Yps[3])
        perp = geomean([perp_hue, perp_sat, perp_val])

        mse_to_peak = mse_from_peak(test_hsv, tuple(Yps...))
    end
end


function laplace_smooth{N}(mdl::TermToColorDistributionEmpirical{N}, train_text)
    lblfreqs = Dict(lbl=>length(inds) for (lbl, inds) in labelmap(train_text))
    
    function smooth(ps, lbl)
        counts = ps*lblfreqs[lbl]
        (counts.+1)./(sum(counts.+1))
    end
    term2dist= Dict(lbl=>tuple((smooth(ps, lbl) for ps in pss)...) for (lbl, pss) in mdl.term2dist)
    
    TermToColorDistributionEmpirical(mdl.output_res, term2dist)
end
