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

const full_cldata = load_monroe_data(dev_as_train=false, dev_as_test=true)

const g_eval_texts = rare_descriptions(full_cldata.train.texts, 100, 8)
const cldata = extrapolation_dataset(full_cldata, g_eval_texts)

@show size(cldata.dev.texts)
const g_output_res = 64

function main(splay_std_dev_in_bins)
    runname = joinpath("extrapolate_validation","sib$(splay_std_dev_in_bins)")
    println("begin $runname")
    datadir = joinpath(Pkg.dir("ColoringNames"), "models", "$runname")
    mkdir(datadir)
    
    extra_data = @names_from begin
        executing_file = @__FILE__
        log_path = joinpath(datadir, "logs")
        mkdir(log_path)

        eval_texts = g_eval_texts

        splay_std_dev_in_bins=splay_std_dev_in_bins
        splay_std_dev = splay_std_dev_in_bins/g_output_res
        epochs = 30
        batch_size = 1_432
    end

    println("initialising $runname network")
    mdl = TermToColorDistributionNetwork(cldata.encoding; batch_size=batch_size)

    println("training $runname network")
    extra_data[:training_costs_o] = train!(mdl,
                                        cldata.train.terms_padded,
                                        cldata.train.colors,
                                        log_path;
                                        splay_stddev=splay_std_dev,
                                        epochs=epochs
                                        )

    println("evaluating $runname")
    extra_data[:dev_set_results] = evaluate(mdl, cldata.dev.terms_padded, cldata.dev.colors)

    println("saving $runname")
    save(mdl, datadir; extra_data...)
end

main(0.5)




















#EOF
