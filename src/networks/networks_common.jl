



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


function restore(::Type{T}, param_path, model_path=load(param_path,"model_path")) where T
    @load(param_path, encoding, max_tokens, hidden_layer_size, embedding_dim, output_res)

    sess, optimizer = init_terms_to_color_dist_network_session(nlabel(encoding), max_tokens, hidden_layer_size, embedding_dim, output_res)
    train.restore(train.Saver(), sess, model_path)

    T(encoding, sess, optimizer,  max_tokens, output_res, hidden_layer_size, embedding_dim)
end



function train!(mdl, cldata, args...; kwargs...)

    train_text, train_terms_padded, train_hsvps =  find_distributions(cldata.train, output_res(mdl))
    train!(mdl, train_text, train_terms_padded, train_hsvps, args...; kwargs...)
end
            

######### ML methods

abstract type AbstractTermToColorDistributionML end
            
            
output_res(mdl::AbstractTermToColorDistributionML) = TensorFlow.get_shape(mdl.sess.graph["Yp_obs_hue"], 1)
n_steps(mdl::AbstractTermToColorDistributionML) = TensorFlow.get_shape(mdl.sess.graph["terms"], 1)

function train!(mdl::AbstractTermToColorDistributionML, train_text, train_terms_padded, train_hsvps::NTuple{3};
                log_dir=nothing,
                batch_size=min(2^14, nobs(train_terms_padded)),
                dropout_keep_prob=0.5f0,
                min_epochs=300,
                max_epochs=30_000,
                early_stopping = ()->0.0   
                )

    ss = mdl.sess.graph
    if log_dir!=nothing
        summary_op = Summaries.merge_all() #FEAR: Does this break if the default graph has changed?
        summary_writer = Summaries.FileWriter(log_dir; graph=ss)
    else
        warn("No log_dir set during training; no logs will be kept.")
    end

    prev_es_loss = early_stopping()
    @progress "Epochs" for epoch_ii in 1:max_epochs

        data = shuffleobs((train_hsvps..., train_terms_padded))
        batchs = eachbatch(data; maxsize=batch_size)
        true_batch_size = floor(nobs(data)/length(batchs))
        if true_batch_size < 0.5*batch_size
            warn("Batch size is only $(true_batch_size)")
        end


        @progress "Batches" for (hp_obs, sp_obs, vp_obs, terms) in batchs
            optimizer_o = run(
                mdl.sess,
                mdl.optimizer,
                Dict(
                    ss["keep_prob"]=>dropout_keep_prob,
                    ss["terms"]=>terms,
                    ss["Yp_obs_hue"]=>hp_obs,
                    ss["Yp_obs_sat"]=>sp_obs,
                    ss["Yp_obs_val"]=>vp_obs
                )
            )

        end

        #Log summary
        if log_dir!=nothing
            (hp_obs, sp_obs, vp_obs, terms) = first(batchs)
            # use the first batch to eval on, the one we trained least recently.  they are shuffled every epoch anyway
            summaries = run(mdl.sess, summary_op,
                    Dict(
                        ss["keep_prob"]=>1.0,
                        ss["terms"]=>terms,
                        ss["Yp_obs_hue"]=>hp_obs,
                        ss["Yp_obs_sat"]=>sp_obs,
                        ss["Yp_obs_val"]=>vp_obs
                    )
                )

            write(summary_writer, summaries, epoch_ii)
        end
                    
        #Early stopping
        if epoch_ii > min_epochs
            es_loss = early_stopping()
            if es_loss > prev_es_loss
                break
            end
            prev_es_loss = es_loss
        end
    end
    mdl
end



function query(mdl,  input_text, encoding=mdl.encoding, max_tokens=n_steps(mdl))
    label = input_text
    labels, _ = prepare_labels([label], encoding, do_demacate=false)

    nsteps_to_pad = max(max_tokens - size(labels,1), 0)

    padded_labels = [labels; zeros(Int, nsteps_to_pad)]
    ss=mdl.sess.graph
    hp, sp, vp = run(
        mdl.sess,
        [
            ss["Yp_hue"],
            ss["Yp_sat"],
            ss["Yp_val"]
        ],
        Dict(
            ss["keep_prob"]=>1.0f0,
            ss["terms"]=>padded_labels,
        )
    )

    hp[1,:], sp[1,:], vp[1,:]
end
            
evaluate(mdl, testdata::ColorDataset) = evaluate(mdl, testdata.texts, testdata.terms_padded, testdata.colors)

"Run all evalutations, returning a dictionary of results"
function evaluate(mdl, test_texts, test_terms_padded, test_hsv)
    gg=mdl.sess.graph

    Y_obs_hue = test_hsv[:, 1]
    Y_obs_sat = test_hsv[:, 2]
    Y_obs_val = test_hsv[:, 3]

    Yp_hue, Yp_sat, Yp_val = run(mdl.sess,
                                 [gg["Yp_hue"], gg["Yp_sat"], gg["Yp_val"]],
                                 Dict(gg["terms"]=>test_terms_padded, gg["keep_prob"]=>1.0))

    @names_from begin
        perp_hue = descretized_perplexity(Y_obs_hue, Yp_hue)
        perp_sat = descretized_perplexity(Y_obs_sat, Yp_sat)
        perp_val = descretized_perplexity(Y_obs_val, Yp_val)
        perp = geomean([perp_hue perp_sat perp_val])

        mse_to_peak = mse_from_peak([Y_obs_hue Y_obs_sat Y_obs_val], (Yp_hue, Yp_sat, Yp_val))
    end
end
