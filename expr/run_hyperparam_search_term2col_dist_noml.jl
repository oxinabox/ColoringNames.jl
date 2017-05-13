using SwiftObjectStores
using ColoringNames
using MLDataPattern
using MLLabelUtils
using StaticArrays
using Juno
using StatsBase
using TensorFlow
using JLD
using FileIO


const serv=SwiftService()

println("loading data")
const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw; do_demacate=false)
const valid_text = valid_raw[:, 1]

#const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
#const train_hsv, train_terms_padded, encoding = prepare_data(train_raw, encoding; do_demacate=false)
#const train_text = train_raw[:, 1]
@show size(valid_hsv)

const train_raw = valid_raw
const train_hsv = valid_hsv
const train_text = valid_text
const train_terms_padded = valid_terms_padded

function main(g_output_res, splay_std_dev_in_bins)
    runname = joinpath("noml_validation","sib$(splay_std_dev_in_bins)_or$(g_output_res)")
    println("begin $runname")
    datadir = joinpath(Pkg.dir("ColoringNames"), "models", "$runname")
    mkdir(datadir)


    println("initialising $runname network")
    extra_data = @names_from begin
        executing_file = @__FILE__
        git_hash = strip(readstring(`git rev-parse --verify HEAD`))
        splay_std_dev_in_bins=splay_std_dev_in_bins
        splay_std_dev = splay_std_dev_in_bins/g_output_res
        output_res=g_output_res
    end

    mdl = TermToColorDistributionEmpirical(g_output_res)

    println("training $runname network")
    train!(mdl, train_text, train_hsv, splay_stddev=splay_std_dev)

    println("evaluating $runname")
    extra_data[:validation_set_results] = evaluate(mdl, valid_text, valid_hsv)

    println("saving $runname")
    extra_data[:model]=mdl
    save(joinpath(datadir, "params_with_model.jld"), stringify_keys(extra_data))
end


for output_res in [32, 64, 128, 256, 512]
    for spread in [32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        gc()
        try
            main(output_res, spread)
        catch ex
            warn(ex)
        end
    end
end



















#EOF
