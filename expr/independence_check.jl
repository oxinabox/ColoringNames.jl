using SwiftObjectStores
using ColoringNames
using Distributions
using MLDataUtils
using Iterators
using MLLabelUtils
using StaticArrays
using Juno
using StatsBase
using Colors
using DataFrames
using Query
using Plots
gr()

const od =(MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.Last())
const serv=SwiftService()

const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw; do_demacate=false)
const valid_text = valid_raw[:, 1]

const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
const train_hsv, train_terms_padded, encoding = prepare_data(train_raw, encoding; do_demacate=false)
const train_text = train_raw[:, 1]




function pairwise_stats(fun, names, hsvs)
    dt = DataFrame(name=String[], n_samples=Int[], hs=Float64[], hv=Float64[], vs=Float64[])
    @progress for (name, inds) in labelmap(names)
        eg_hsvs = @view hsvs[inds, :]

        hs = fun(eg_hsvs[:,1], eg_hsvs[:,2])
        hv = fun(eg_hsvs[:,1], eg_hsvs[:,3])
        vs = fun(eg_hsvs[:,3], eg_hsvs[:,2])

        push!(dt, [name, length(inds), hs, hv, vs])
    end
    dt
end
spearman = pairwise_stats(corspearman, train_text, train_hsv)
spearman_pla

histogram(([spearman[:hs],
            spearman[:hv],
            spearman[:vs],
            maximum(abs.(Matrix(spearman[[:hs, :hv, :vs]])), 2)[:]]),
            layout=(4,1))


@which describe(spearman)


high_cor =
    @from r in spearman begin
    @where abs(r.hs) > 0.30 || abs(r.hv) > 0.30 || abs(r.vs) > 0.30
    @orderby abs(r.hs) + abs(r.hv) + abs(r.vs)
    @select r
    @collect DataFrame
end





dt = DataFrame(a=rand(10), b=randn(10))
describe(dt)


spearman[:vs]
















#EOF
