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

const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
const train_hsv, train_terms_padded, encoding = prepare_data(train_raw, encoding; do_demacate=false)
const train_text = train_raw[:, 1]

#const train_raw = valid_raw
#const train_hsv = valid_hsv
#const train_text = valid_text
#const train_terms_padded = valid_terms_padded

const g_output_res = 64
const g_splay_stddev=1/g_output_res

println("splaying probabilities")
const train_hsvps = splay_probabilities(train_hsv, g_output_res, g_splay_stddev)


function main(embedding_dim, hidden_layer_size)
    runname = joinpath("hyperparam_validation","emb$(embedding_dim)_hl$(hidden_layer_size)_or$(g_output_res)")
#    runname = joinpath("checkruns_$(embedding_dim)_hl$(hidden_layer_size)_or$(g_output_res)")
    println("begin $runname")
    datadir = joinpath(Pkg.dir("ColoringNames"), "models", "$runname")
    mkdir(datadir)

    run_data = @names_from begin
        executing_file = @__FILE__
        git_hash = strip(readstring(`git rev-parse --verify HEAD`))


        model_path = joinpath(datadir, "model.jld")
        meta_path = joinpath(datadir, "meta.jld")
        log_path = joinpath(datadir, "logs")
        mkdir(log_path)

        batch_size = 12_381 #Factors of the true test set size are: 2 * 2 * 3 * 11 * 4_127 = 5_44_764
        output_res = g_output_res
        n_steps=size(valid_terms_padded,1)
        n_classes = nlabel(encoding)+1

        hidden_layer_size = hidden_layer_size
        embedding_dim = embedding_dim

        learning_rate = 0.5
        epochs=100
        splay_stddev=g_splay_stddev
    end

    println("initialising $runname network")
    sess, optimizer = terms_to_color_dist_network(
                                                n_classes,
                                                n_steps;
                                                output_res = output_res,
                                                batch_size = batch_size,
                                                embedding_dim = embedding_dim,
                                                hidden_layer_size = hidden_layer_size,
                                                learning_rate = learning_rate)


    println("training $runname network")
    run_data[:training_costs_o] = train_to_color_dist!(
                                                    sess,
                                                    optimizer,
                                                    batch_size,
                                                    output_res,
                                                    train_terms_padded,
                                                    train_hsvps,
                                                    log_path;
                                                    epochs=epochs
                                                    )

    println("evaluating $runname")
    run_data[:results] = evaluate(sess, batch_size, valid_terms_padded, valid_hsv)

    println("saving $runname")
    train.save(train.Saver(), sess, model_path)
    save(meta_path, stringify_keys(run_data))
end

for emb in [3, 16, 32, 64], hl in [32, 64, 128, 256]
    gc()
    try
        main(emb, hl)
    catch ex
        warn(ex)
    end
end




















#EOF
