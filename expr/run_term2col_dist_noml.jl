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



println("loading data")


const cldata = load_monroe_data(dev_as_train=false, dev_as_test=true)


function main(g_output_res, splay_std_dev_in_bins)
    runname = joinpath("noml","sib$(splay_std_dev_in_bins)_or$(g_output_res)")
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
    train!(mdl, cldata.train.texts, cldata.train.colors, splay_stddev=splay_std_dev)
    extra_data[:model]=mdl

    println("evaluating $runname")
    extra_data[:validation_set_results] = evaluate(mdl, cldata.dev.texts, cldata.dev.colors)

    println("saving $runname")
    save(joinpath(datadir, "emprical_model.jld"), stringify_keys(extra_data))
##########################################
    println("Smoothing")
    #Overwrite model with smooth one -- the rest is the same.
    mdl = laplace_smooth(mdl, cldata.train.texts)
    extra_data[:mdl]=mdl
    extra_data[:validation_set_results] = evaluate(mdl, cldata.dev.texts, cldata.dev.colors)
    save(joinpath(datadir, "smoothed_emprical_model.jld"), stringify_keys(extra_data))
end


for output_res in [64,256]
    for spread in [0.5]
        gc()
        try
            main(output_res, spread)
        catch ex
            warn(ex)
        end
    end
end





#EOF
