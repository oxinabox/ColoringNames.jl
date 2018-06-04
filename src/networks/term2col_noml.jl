const Summaries = TensorFlow.summary

mutable struct TermToColorDistributionEmpirical <: AbstractDistEstModel
    encoding::LabelEnc.NativeLabels
    output_res::Int
    hsvp::NTuple{3, Array{Float64,2}}
    function TermToColorDistributionEmpirical(output_res=256)
        ret = new()
        ret.output_res = output_res
        ret
    end
end

"Dummy constructor for compatibility"
function TermToColorDistributionEmpirical(args...; output_res=256, kwargs...)
    TermToColorDistributionEmpirical(output_res)
end




output_res(mdl::TermToColorDistributionEmpirical) = mdl.output_res

float_type(mdl::TermToColorDistributionEmpirical) = Float64
function train!(mdl::TermToColorDistributionEmpirical, train_text, train_terms_padded, train_hsvps::NTuple{3};
    remove_zeros_hack = true, kwargs...
    )
    mdl.encoding = labelenc(train_text)
    mdl.hsvp = train_hsvps
    if remove_zeros_hack
        # Adding eps to every single probability technically breaks probability
        # by causing the estimates to not sum to 1
        # But it is by an amount that is smaller than the that caused by using Float32s
        # In the outputs of the other models.
        mdl.hsvp[1].+=eps()
        mdl.hsvp[2].+=eps()
        mdl.hsvp[3].+=eps()
    end
    mdl
end


function query(mdl::TermToColorDistributionEmpirical,  input_texts::AbstractVector, args...)
    ind = convertlabel(LabelEnc.Indices, String.(input_texts), mdl.encoding)
    mdl.hsvp[1][:,ind], mdl.hsvp[2][:, ind], mdl.hsvp[3][:, ind]
end


######################################################################


mutable struct TermToColorPointEmpirical <: AbstractPointEstModel
    encoding::LabelEnc.NativeLabels
    hsv::Array{Float32,2}
    TermToColorPointEmpirical() = new()
end

TermToColorPointEmpirical(args...; kwargs...) = TermToColorPointEmpirical() # dummy constructor for compat


function train!(mdl::TermToColorPointEmpirical, train_text, train_terms_padded, train_hsvs::Matrix; kwargs...)
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
    mdl.hsv = hsv
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
    h = mod(atan2(sh, ch), 2π)/2π
    
    (h, s, v)
end


function query(mdl::TermToColorPointEmpirical,  input_text::AbstractVector)
    ind = convertlabel(LabelEnc.Indices, String.(input_text), mdl.encoding)
    mdl.hsv[:, ind]
end
