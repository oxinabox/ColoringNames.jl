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

ColoringNames.@load_monroe_data()

const g_output_res = 256

function main(splay_std_dev_in_bins)
    runname = joinpath("wide_or_hyperparam_validation","sib$(splay_std_dev_in_bins)")
    println("begin $runname")
    datadir = joinpath(Pkg.dir("ColoringNames"), "models", "$runname")
    mkdir(datadir)



    extra_data = @names_from begin
        executing_file = @__FILE__
        log_path = joinpath(datadir, "logs")
        mkdir(log_path)

        splay_std_dev_in_bins=splay_std_dev_in_bins
        splay_std_dev = splay_std_dev_in_bins/g_output_res
        epochs = 30
    end

    println("initialising $runname network")
    mdl = TermToColorDistributionNetwork(encoding)

    println("training $runname network")
    extra_data[:training_costs_o] = train!(mdl,
                                        train_terms_padded,
                                        train_hsv,
                                        log_path;
                                        splay_stddev=splay_std_dev,
                                        epochs=epochs
                                        )

    println("evaluating $runname")
    extra_data[:validation_set_results] = evaluate(mdl, valid_terms_padded, valid_hsv)

    println("saving $runname")
    save(mdl, datadir; extra_data...)
end

for spread in [32, 16, 8, 4, 2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    gc()
    try
        main(spread)
    catch ex
        warn(ex)
    end
end




















#EOF
