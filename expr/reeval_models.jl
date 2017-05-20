using SwiftObjectStores
using ColoringNames
using MLDataPattern
using MLLabelUtils
using StaticArrays
using Juno
using StatsBase
using TensorFlow
using JLD
using Glob

const serv=SwiftService()

println("loading data")
const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw; do_demacate=false)
const valid_text = valid_raw[:, 1]

function main(orig_meta_path)
    println("begin $(orig_meta_path)")
    run_data = load(orig_meta_path)

    run_data["executing_file_eval"] = @__FILE__
    run_data["git_hash_evaltime"] = strip(readstring(`git rev-parse --verify HEAD`))


    run_data["meta_path"] = joinpath(dirname(orig_meta_path), "meta_v2.jld")
    println("initialising network")
    sess, optimizer = terms_to_color_dist_network(
                                                run_data["n_classes"],
                                                run_data["n_steps"];
                                                output_res = run_data["output_res"],
                                                batch_size = run_data["batch_size"],
                                                embedding_dim = run_data["embedding_dim"],
                                                hidden_layer_size = run_data["hidden_layer_size"],
                                                learning_rate = run_data["learning_rate"])


    println("loading network")
    train.restore(train.Saver(), sess, run_data["model_path"])

    println("evaluating")
    delete!(run_data, "results")
    run_data["results_validation_set"] = evaluate(sess, run_data["batch_size"], valid_terms_padded, valid_hsv)

    println("saving")
    save(run_data["meta_path"], stringify_keys(run_data))
end

cd(Pkg.dir("ColoringNames"))
for fn in  glob(glob"models/hyperparam_validation/*/meta.jld")
    main(fn)
end



















#EOF
