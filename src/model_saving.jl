
function FileIO.save(mdl, save_dir; extra_info...)
    params = Dict(string(nn)=>getfield(mdl,nn) for nn in fieldnames(mdl) if !(nn in [:sess, :optimizer]))
    for (kk, vv) in extra_info
        params[string(kk)] = vv
    end
    params["save_time"] = now()
    params["git_hash"] = strip(readstring(`git rev-parse --verify HEAD`))

    params["model_path"] = joinpath(save_dir, "model.jld")
    save(joinpath(save_dir, "params.jld"), params)

    train.save(train.Saver(), mdl.sess, params["model_path"])
end


function restore(::Type{T}, param_path, model_path=load(param_path,"model_path"))
    @load(param_path, encoding, max_tokens, hidden_layer_size, embedding_dim, output_res)

    sess, optimizer = init_terms_to_color_dist_network_session(nlabel(encoding), max_tokens, hidden_layer_size, embedding_dim, output_res)
    train.restore(train.Saver(), sess, model_path)

    T(encoding, sess, optimizer,  max_tokens, output_res, hidden_layer_size, embedding_dim)
end
