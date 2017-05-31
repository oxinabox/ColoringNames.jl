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

const g_output_res = 64

function main(name, splay_std_dev_in_bins)
    runname = joinpath(name,"$(now())_sib$(splay_std_dev_in_bins)")
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
    mdl = TermToColorDistributionNetwork(cldata.encoding)

    println("training $runname network")
    extra_data[:training_costs_o] = train!(mdl,
                                        cldata.train.terms_padded,
                                        cldata.train.colors,
                                        log_path;
                                        splay_stddev=splay_std_dev,
                                        epochs=epochs
                                        )

    println("evaluating $runname")
    extra_data[:validation_set_results] = evaluate(mdl, cldata.test.terms_padded, cldata.test.colors)

    println("saving $runname")
    save(mdl, datadir; extra_data...)
end

main("good", 0.5)




#EOF
