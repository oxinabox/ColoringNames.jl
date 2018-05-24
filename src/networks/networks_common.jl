abstract type AbstractDistEstModel end
abstract type AbstractPointEstModel end

abstract type AbstractDistEstML  <: AbstractDistEstModel end
abstract type AbstractPointEstML <: AbstractPointEstModel end
const AbstractModelML = Union{AbstractDistEstML, AbstractPointEstML}



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

function train!(mdl, cldata::ColorDatasets, args...; kwargs...)
    df_kwargs = default_train_kwargs(mdl, cldata,  args...)
    data = (cldata.train.texts, cldata.train.terms_padded, cldata.train.colors)
    train!(mdl, data..., args...; df_kwargs..., kwargs...)
end
                
function train!(mdl, cldata::ColorDatasets, smoothing, args...; kwargs...)
    dists =  find_distributions(cldata.train, output_res(mdl), smoothing, float_type(mdl))
    df_kwargs = default_train_kwargs(mdl, cldata, smoothing, args...)
    train!(mdl, dists..., args...; df_kwargs..., kwargs...)
end
            



function default_train_kwargs(mdl, cldata, args...)
    Dict()
end
            
evaluate(mdl, testdata::ColorDataset) = evaluate(mdl, testdata.texts, testdata.terms_padded, testdata.colors)



"Run all evalutations, returning a dictionary of results"
function evaluate(mdl::AbstractDistEstModel, test_texts, test_terms_padded, test_hsv)
                
    Y_obs_hue = @view(test_hsv[:, 1])
    Y_obs_sat = @view(test_hsv[:, 2])
    Y_obs_val = @view(test_hsv[:, 3])

    Yp_hue, Yp_sat, Yp_val = query(mdl, test_texts)

    @names_from begin
        perp_hue = descretized_perplexity(Y_obs_hue, Yp_hue')
        perp_sat = descretized_perplexity(Y_obs_sat, Yp_sat')
        perp_val = descretized_perplexity(Y_obs_val, Yp_val')
        perp = geomean([perp_hue perp_sat perp_val])

        mse_to_peak = mse_from_peak([Y_obs_hue Y_obs_sat Y_obs_val], (Yp_hue', Yp_sat', Yp_val'))
    end
end

            
function evaluate(mdl::AbstractPointEstModel, texts, terms_padded, reference_colors)
    preds = query(mdl, texts)'
    @assert(size(reference_colors) == size(preds), "$(size(reference_colors)) != $(size(preds))")
    @assert size(preds,2) == 3
    mse(reference_colors, preds)
end

            
function query(mdl, input_text)
    hp, sp, vp = query(mdl,  [input_text])
    hp[:, 1], sp[:, 1], vp[:, 1]
end

function query(mdl, input_texts::Vector) 
                #should overload this or else will get overflows
    throw(MethodError(query, (typeof(mdl), typeof(input_texts))))
end
#######################################
# AbstractML
             
function query(mdl::AbstractModelML, input_texts::Vector) 
                #should overload this or else will get overflows
    encoding=mdl.encoding
    max_tokens=n_steps(mdl)
                
    labels, _ = prepare_labels(input_texts, encoding, do_demacate=false)

    nsteps_to_pad = max(max_tokens - size(labels,1), 0)
    padded_labels = [labels; zeros(Int, (nsteps_to_pad, length(input_texts)))]
    _query(mdl, padded_labels)
end

n_steps(mdl::AbstractModelML) = TensorFlow.get_shape(mdl.sess.graph["terms"], 1)
            
######### AbstractDistEstML methods           
output_res(mdl::AbstractDistEstML) = TensorFlow.get_shape(mdl.sess.graph["Yp_obs_hue"], 1)


function train!(mdl::AbstractDistEstML, train_text, train_terms_padded, train_hsvps::NTuple{3};
                log_dir=nothing,
                batch_size=min(2^14, nobs(train_terms_padded)),
                dropout_keep_prob=0.5f0,
                min_epochs=0,
                max_epochs=30_000,
                early_stopping = ()->0.0,
                check_freq = 25
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
        data = shuffleobs((train_hsvps..., train_terms_padded))
        batchs = eachbatch(data; maxsize=batch_size)
        true_batch_size = floor(nobs(data)/length(batchs))
        if true_batch_size < 0.5*batch_size
            warn("Batch size is only $(true_batch_size)")
        end

        
        # Each Batch
        for (hp_obs, sp_obs, vp_obs, terms) in batchs
            run(
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
                    
        if epoch_ii % check_freq == 1            
            # Early stopping
            es_loss = early_stopping()           
            epoch_ii > min_epochs && es_loss > prev_es_loss && break
            prev_es_loss = es_loss

            # Log summary
            if log_dir!=nothing
                (hp_obs, sp_obs, vp_obs, terms) = first(batchs)
                # use the first batch to eval on, the one we trained least recently.  they are shuffled every epoch anyway
                summaries = run(mdl.sess, mdl.summary,
                        Dict(
                            ss["keep_prob"]=>1.0,
                            ss["terms"]=>terms,
                            ss["Yp_obs_hue"]=>hp_obs,
                            ss["Yp_obs_sat"]=>sp_obs,
                            ss["Yp_obs_val"]=>vp_obs,
                            ss["early_stopping_loss"] => es_loss
                        )
                    )
                write(summary_writer, summaries, epoch_ii)
            end
        end
    end
    mdl
end


            
"Inner functon for query"
function _query(mdl::AbstractDistEstML,  padded_labels::Matrix)
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
    hp', sp', vp'
end
            

            

            
#######################

            
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
            
##############################
# Dist Est ML            

            
function default_train_kwargs(mdl::AbstractDistEstML, cldata, args...)
    Dict(:early_stopping => () ->evaluate(mdl, cldata.dev)[:perp])
end
          
function init_dist_est_network(combine_timesteps, word_vecs, n_steps, hidden_layer_size, output_res, learning_rate)
    function produce_output(Z)
        @tf begin
            function declare_output_layer(name)
                W = get_variable("W_$name", (hidden_layer_size, output_res), Float32)
                B = get_variable("B_$name", (output_res), Float32)
                Y_logit = identity(Z*W + B, name="Yp_logit_$name")
                Y = nn.softmax(Y_logit; name="Yp_$name")
                Yp_obs = placeholder(Float32; shape=[output_res, -1], name="Yp_obs_$name")'
                loss = nn.softmax_cross_entropy_with_logits(;labels=Yp_obs, logits=Y_logit, name="loss_$name")

                Summaries.scalar("loss_$name", reduce_mean(loss); name="summary_loss_$name")
                # Summaries.histogram("W_$name", W; name="summary_W_$name")
                loss
            end
            loss_hue = declare_output_layer("hue")
            loss_sat = declare_output_layer("sat")
            loss_val = declare_output_layer("val")


            loss_total = reduce_mean(loss_hue + loss_sat + loss_val)
            loss_total
        end
    end
                
    init_ML_network(combine_timesteps, produce_output, word_vecs, n_steps, learning_rate)
end
            
            
#######################
# Point Est network ]) end
            
            
            
function default_train_kwargs(mdl::AbstractPointEstML, cldata, args...)
    Dict(:early_stopping => () ->evaluate(mdl, cldata.dev))
end
          
function init_point_est_network(combine_timesteps, word_vecs, n_steps, hidden_layer_size, learning_rate)
    function produce_output(Z)
        @tf begin
            #=    
                Y_logit = identity(Z*W + B, name="Yp_logit_$name")
                Y = nn.softmax(Y_logit; name="Yp_$name")
                Yp_obs = placeholder(Float32; shape=[output_res, -1], name="Yp_obs_$name")'
                loss = nn.softmax_cross_entropy_with_logits(;labels=Yp_obs, logits=Y_logit, name="loss_$name")

                Summaries.scalar("loss_$name", reduce_mean(loss); name="summary_loss_$name")
                # Summaries.histogram("W_$name", W; name="summary_W_$name")
                loss
                        end =#
            Wo = get_variable((hidden_layer_size, 4), Float32)
            bo = get_variable(4, Float32)
            Y_logit = identity(Z*Wo + bo)
            Y_hue = atan2(Y_logit[:,1],Y_logit[:,2])/(2Float32(π))
            Y_sat = nn.sigmoid(Y_logit[:,3])
            Y_val = nn.sigmoid(Y_logit[:,4])
            Y = identity([Y_hue Y_sat Y_val])
            @show Y
            
            Y_obs = placeholder(Float32; shape=[-1 ,3])
            loss_total = mse(Y, Y_obs)
            @show loss_total
            loss_total
        end
    end
                
    init_ML_network(combine_timesteps, produce_output, word_vecs, n_steps, learning_rate)
end
            
            
            
function _query(mdl::AbstractPointEstML,  padded_labels::Matrix)
    ss=mdl.sess.graph
    hsv = run(
        mdl.sess,
        ss["Y"],
        Dict(
            ss["keep_prob"]=>1.0f0,
            ss["terms"]=>padded_labels,
        )
    )
    hsv'
end