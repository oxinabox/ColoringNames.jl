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

const cldata = load_monroe_data(dev_as_train=false, dev_as_test=true)


@show size(cldata.dev.texts)
const g_output_res = 256

function main(splay_std_dev_in_bins)
    runname = joinpath("highdim","sib$(splay_std_dev_in_bins)")
    println("begin $runname")
    datadir = joinpath(Pkg.dir("ColoringNames"), "models", "$runname")
    mkdir(datadir)
    
    extra_data = @names_from begin
        executing_file = @__FILE__
        log_path = joinpath(datadir, "logs")
        mkdir(log_path)


        splay_std_dev = splay_std_dev_in_bins/g_output_res
        epochs = 50
        batch_size = 1_000
    end

    println("initialising $runname network")
    mdl = TermToColorDistributionNetwork(cldata.encoding; output_res=g_output_res)

    println("training $runname network")
    train!(mdl,
            cldata.train.terms_padded,
            cldata.train.colors,
            log_path;
            batch_size = batch_size,
            splay_stddev=splay_std_dev,
            epochs=epochs
            )

    println("Saving pre_eval, $runname")
    preeval_dir = joinpath(datadir,"pre_eval")
    mkdir(preeval_dir)
    save(mdl, preeval_dir; extra_data...)

    println("evaluating $runname")
    extra_data[:dev_set_results] = evaluate(mdl, cldata.dev.terms_padded, cldata.dev.colors)

    println("saving $runname")
    save(mdl, datadir; extra_data...)
    rm(preeval_dir; recursive=true)
end

for var in [0.5, 1, 2, 4, 0.25, 0.125]
    gc()
    try
        main(var)
    catch ex
        warn(ex)
    end
end


#EOF
