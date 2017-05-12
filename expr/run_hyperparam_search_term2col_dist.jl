using SwiftObjectStores
using ColoringNames
using MLDataPattern
using MLLabelUtils
using StaticArrays
using Juno
using StatsBase
using TensorFlow
using JLD


const od =(ObsDim.First(), ObsDim.Last())

const serv=SwiftService()

println("loading data")
const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw; do_demacate=false)
const valid_text = valid_raw[:, 1]

#const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
#const train_hsv, train_terms_padded, encoding = prepare_data(train_raw, encoding; do_demacate=false)
#const train_text = train_raw[:, 1]

const train_raw = valid_raw
const train_hsv = valid_hsv
const train_text = valid_text
const train_terms_padded = valid_terms_padded

const g_output_res = 64

function main(splay_std_dev_in_bins)
    runname = joinpath("spread_validation","sib$(splay_std_dev_in_bins)")
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
    run_data[:training_costs_o] = train_to_color_dist!(mdl,
                                                    train_terms_padded,
                                                    train_hsvps,
                                                    log_path;
                                                    epochs=epochs
                                                    )

    println("evaluating $runname")
    run_data[:validation_set_results] = evaluate(mdl, valid_terms_padded, valid_hsv)

    println("saving $runname")
    save(meta_path, stringify_keys(run_data))
end

for spread in [3]
    gc()
#    try
        main(spread)
#    catch ex
#        warn(ex)
#    end
end




















#EOF
