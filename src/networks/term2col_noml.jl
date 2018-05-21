const Summaries = TensorFlow.summary

mutable struct TermToColorDistributionEmpirical
    encoding::LabelEnc.NativeLabels
    output_res::Int
    hsvp::NTuple{3, Array{Float32,2}}
    function TermToColorDistributionEmpirical(output_res=64)
        ret = new()
        ret.output_res = output_res
        ret
    end
end

output_res(mdl::TermToColorDistributionEmpirical) = mdl.output_res


function train!(mdl::TermToColorDistributionEmpirical, train_text, train_terms_padded, train_hsvps::NTuple{3})
    mdl.encoding = labelenc(train_text)
    mdl.hsvp = train_hsvps
end


function query(mdl::TermToColorDistributionEmpirical,  input_text)
    ind = convertlabel(LabelEnc.Indices, input_text, mdl.encoding)
    mdl.hsvp[1][:,ind], mdl.hsvp[2][:, ind], mdl.hsvp[3][:, ind]
end


"Run all evalutations, returning a dictionary of results"
function evaluate(mdl::TermToColorDistributionEmpirical, test_texts, test_terms_padded, test_hsv)
    N = output_res(mdl)
    Yps = [Matrix{Float32}(length(test_texts), mdl.output_res) for ii in 1:N]
    
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


function laplace_smooth!(mdl::TermToColorDistributionEmpirical, train_text)
    lblfreqs = Dict(lbl=>length(inds) for (lbl, inds) in labelmap(train_text))
    
    function smooth(ps, lbl)
        counts = ps*lblfreqs[lbl]
        (counts.+1)./(sum(counts.+1))
    end
    term2dist= Dict(lbl=>tuple((smooth(ps, lbl) for ps in pss)...) for (lbl, pss) in mdl.term2dist)
    
    mdl
end


######################################################################


mutable struct TermToColorPointEmpirical
    encoding::LabelEnc.NativeLabels
    output_res::Int
    hsv::Array{Float32,2}
    function TermToColorPointEmpirical(output_res=64)
        ret = new()
        ret.output_res = output_res
        ret
    end
end


function train!(mdl::TermToColorPointEmpirical, train_text, train_terms_padded, train_hsvs::Matrix)
    grps = collect(groupby(last, enumerate(train_text)))
    len = length(grps)
    
    texts = Vector{String}(len)
    hsv = Matrix{Float32}(3, len)


    for (ii, grp) in enumerate(grps)
        ind_start, texts[ii] = first(grp)
        ind_end  = first(last(grp))

        inds = ind_start:ind_end # faster to slice with ranges
        colors = @view train_hsvs[inds, :]
        
        hsv[:, ii] .= hsv_mean(colors)
    end
    
    mdl.encoding = labelenc(texts)
    mdl.hsvp = hsv
    mdl
end

"""
Angularly correct mean of HSV values.
Not (nesc) perceptual mean.
"""
function hsv_mean(colors)
    hs = @view(colors[:,1])
    s = mean(@view(colors[:,2]))
    v = mean(@view(colors[:,3]))
    
    ch = mean(cos.(2π*hs))
    sh = mean(sin.(2π*hs))
    h = atan2(sh, ch)/2π
    
    (h, s, v)
end


function query(mdl::TermToColorPointEmpirical,  input_text)
    ind = convertlabel(LabelEnc.Indices, input_text, mdl.encoding)
    mdl.hsv[:, ind]
end
