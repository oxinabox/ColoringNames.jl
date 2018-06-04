


function train!(mdl::AbstractDistEstModel, cldata::ColorDatasets, smoothing, args...; kwargs...)
    dists =  find_distributions(cldata.train, output_res(mdl), smoothing, float_type(mdl))
    df_kwargs = default_train_kwargs(mdl, cldata, smoothing, args...)
    train!(mdl, dists..., args...; df_kwargs..., kwargs...)
end


function query(mdl::AbstractDistEstModel, input_text)
    hp, sp, vp = query(mdl,  [input_text])
    hp[:, 1], sp[:, 1], vp[:, 1]
end


"Run all evalutations, returning a dictionary of results"
function evaluate(mdl::AbstractDistEstModel, test_texts, test_terms_padded, test_hsv)
                
    Y_obs_hue = @view(test_hsv[:, 1])
    Y_obs_sat = @view(test_hsv[:, 2])
    Y_obs_val = @view(test_hsv[:, 3])
    Y_obs = [Y_obs_hue Y_obs_sat Y_obs_val]
                
    Yp_hue, Yp_sat, Yp_val = map(transpose, query(mdl, test_texts))

                       
    Dict{Symbol, Any}(
        :perp_hue => descretized_perplexity(Y_obs_hue, Yp_hue),
        :perp_sat => descretized_perplexity(Y_obs_sat, Yp_sat),
        :perp_val => descretized_perplexity(Y_obs_val, Yp_val),
      
        :perp => full3d_descretized_perplexity((Y_obs_hue, Y_obs_sat, Y_obs_val),(Yp_hue, Yp_sat, Yp_val)),

        :mse_to_distmode => mse_from_peak(Y_obs, (Yp_hue, Yp_sat, Yp_val)),
        :mse_to_distmean => mse_from_distmean(Y_obs, (Yp_hue, Yp_sat, Yp_val)),
    )
end


######### AbstractDistEstML methods           
output_res(mdl::AbstractDistEstML) = TensorFlow.get_shape(mdl.sess.graph["Yp_obs_hue"], 1)
            
function default_train_kwargs(mdl::AbstractDistEstML, cldata, args...)
    Dict(:early_stopping => () ->evaluate(mdl, cldata.dev)[:perp])
end
          

function train!(mdl::AbstractDistEstML, train_text, train_terms_padded, train_hsvps::NTuple{3}; kwargs...)
    ss=mdl.sess.graph
    function obs_input_func(batch)   
        hp_obs, sp_obs, vp_obs, terms = batch
        (
        ss["terms"]=>terms,
        ss["Yp_obs_hue"]=>hp_obs,
        ss["Yp_obs_sat"]=>sp_obs,
        ss["Yp_obs_val"]=>vp_obs
        )
    end
    
                
    _train!(obs_input_func, mdl, (train_hsvps..., train_terms_padded);
        batch_size=nobs(train_terms_padded),
        kwargs...)               
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