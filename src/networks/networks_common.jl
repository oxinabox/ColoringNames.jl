

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


float_type(mdl)=Float32





function default_train_kwargs(mdl, cldata, args...)
    Dict()
end
            
evaluate(mdl, testdata::ColorDataset) = evaluate(mdl, testdata.texts, testdata.terms_padded, testdata.colors)

            


function query(mdl, input_texts::AbstractVector) 
                #should overload this or else will get overflows
    throw(MethodError(query, (typeof(mdl), typeof(input_texts))))
end
#######################################
# AbstractML
             
function query(mdl::AbstractModelML, input_texts::AbstractVector) 
                #should overload this or else will get overflows
    encoding=mdl.encoding
    max_tokens=n_steps(mdl)
                
    labels, _ = prepare_labels(input_texts, encoding)

    nsteps_to_pad = max(max_tokens - size(labels,1), 0)
    padded_labels = [labels; zeros(Int, (nsteps_to_pad, length(input_texts)))]
    _query(mdl, padded_labels)
end

n_steps(mdl::AbstractModelML) = TensorFlow.get_shape(mdl.sess.graph["terms"], 1)



function _train!(obs_input_func::Function, mdl, all_obs;
                obsdims=ObsDim.Last(),
                log_dir=nothing,
                batch_size=2^14, # last obs is terms
                dropout_keep_prob=0.5f0,
                min_epochs=0,
                max_epochs=30_000,
                early_stopping = ()->0.0,
                check_freq = 10
                )

    ss = mdl.sess.graph
    if log_dir!=nothing
        summary_writer = Summaries.FileWriter(log_dir; graph=ss)
    else
        warn("No log_dir set during training; no logs will be kept.")
    end

    prev_es_loss = early_stopping()
    for epoch_ii in 1:max_epochs

        # Setup Batches
        data = shuffleobs(all_obs, obsdims)
        batchs = eachbatch(data; obsdim=obsdims, size=min(batch_size, nobs(data, obsdims)))
                
        # Each Batch
        for batch in batchs
            run(
                mdl.sess,
                mdl.optimizer,
                Dict(
                    ss["keep_prob"]=>dropout_keep_prob,
                    obs_input_func(batch)...
                )
            )

        end
                    
        if epoch_ii % check_freq == 1 
            # Early stopping
            es_loss = early_stopping()
            epoch_ii > min_epochs && es_loss > prev_es_loss && break
            prev_es_loss = es_loss

            # Log summary
            if log_dir!=nothing
                # use the first batch to eval on, the one we trained least recently.  they are shuffled every epoch anyway
                summaries = run(mdl.sess, mdl.summary,
                        Dict(
                            ss["keep_prob"]=>1.0,
                            ss["early_stopping_loss"] => es_loss,
                            obs_input_func(first(batchs))...
                        )
                    )
                write(summary_writer, summaries, epoch_ii)
            end
        end
    end
    mdl
end
            
            
            
function init_ML_network(combine_timesteps, produce_output, word_vecs, n_steps, learning_rate)
    graph = Graph()
    sess = Session(graph)
    @tf begin
        keep_prob = placeholder(Float32; shape=[])
        terms = placeholder(Int32; shape=[n_steps, -1])

        emb_table = [zeros(Float32, size(word_vecs,1))'; word_vecs'] # add an extra-first row for padding
        terms_emb = gather(emb_table, terms+1) # move past the first row we added for zeros)
        ######################## THE ADAPTABLE PART###############
        Z = identity(combine_timesteps(terms_emb, keep_prob))
        cost = identity(produce_output(Z))
        ##########################################################
                    
        optimizer = train.minimize(train.AdamOptimizer(learning_rate), cost)

        summary_cost = Summaries.scalar("cost", cost)
#        summary_W1 = Summaries.histogram("W1", W1)
        
        #External Input for logging
        early_stopping_loss = placeholder(Float32; shape=[]) 
        Summaries.scalar("early_stopping_loss", early_stopping_loss; name="summary_early_stopping_loss")
        summary_op = Summaries.merge_all()
    end
    run(sess, global_variables_initializer())
    sess, optimizer, summary_op   
end
            

            

            
            
#################
            
include("networks_dist_est.jl")
include("networks_point_est.jl")