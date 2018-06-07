

function train!(mdl::AbstractPointEstModel, cldata::ColorDatasets, args...; kwargs...)
    df_kwargs = default_train_kwargs(mdl, cldata,  args...)
    data = (cldata.train.texts, cldata.train.terms_padded, cldata.train.colors)
    train!(mdl, data..., args...; df_kwargs..., kwargs...)
end


function evaluate(mdl::AbstractPointEstModel, texts, terms_padded, reference_colors)
    preds = query(mdl, texts)'
    @assert(size(reference_colors) == size(preds), "$(size(reference_colors)) != $(size(preds))")
    @assert size(preds,2) == 3
    mse(reference_colors, preds)
end


#######################
# Point Est ML network
            
            
            
function default_train_kwargs(mdl::AbstractPointEstML, cldata, args...)
    Dict(:early_stopping => () ->evaluate(mdl, cldata.dev))
end
            
function train!(mdl::AbstractPointEstML, train_text, train_terms_padded, train_hsvs::AbstractMatrix; kwargs...)
    ss=mdl.sess.graph
    function obs_input_func(batch)
        hsv_obs, terms = batch       
        (ss["terms"]=>terms, ss["Y_obs"]=>hsv_obs)
    end
                
    _train!(obs_input_func, mdl,
            (train_hsvs, train_terms_padded); 
            obsdims=(ObsDim.First(), ObsDim.Last()),
            kwargs...)               
end

          
function init_point_est_network(combine_timesteps, word_vecs, n_steps, hidden_layer_size, learning_rate)
    function produce_output(Z)
        @tf begin
            Wo = get_variable((hidden_layer_size, 4), Float32)
            bo = get_variable(4, Float32)
            Y_logit = identity(Z*Wo + bo)
            
            
            Y_sat = nn.sigmoid(Y_logit[:,3])
            Y_val = nn.sigmoid(Y_logit[:,4])
                        
            Y_hue_sin = tanh(Y_logit[:,1])
            Y_hue_cos = tanh(Y_logit[:,2])
                        
            # Obs 
            Y_obs = placeholder(Float32; shape=[-1, 3])
            Y_obs_hue = Y_obs[:, 1]                                
            Y_obs_sat = Y_obs[:, 2]
            Y_obs_val = Y_obs[:, 3]
                        
            Y_obs_hue_sin = sin(Float32(2π).*Y_obs_hue)
            Y_obs_hue_cos = cos(Float32(2π).*Y_obs_hue)
                        
            ## Loss            
                
            loss_hue = reduce_mean(0.5((Y_hue_sin-Y_obs_hue_sin)^2 + (Y_hue_cos-Y_obs_hue_cos)^2))
            Summaries.scalar("loss_hue", loss_hue; name="summary_loss_hue")
                        
            loss_sat = reduce_mean((Y_sat-Y_obs_sat)^2)
            Summaries.scalar("loss_sat", loss_hue; name="summary_loss_sat")
                        
            loss_val = reduce_mean((Y_val-Y_obs_val)^2)
            Summaries.scalar("loss_val", loss_hue; name="summary_loss_val")
            
            
            loss_total = identity(loss_hue + loss_sat + loss_val)
            
                        
            ## For output            
            Y_hue_o1 = Ops.atan2(Y_hue_sin, Y_hue_cos)/(2Float32(π))
            Y_hue_o2 = select(Y_hue_o1 > 0, Y_hue_o1, Y_hue_o1+1) # Wrap around things below 0
            Y_hue = reshape(Y_hue_o2, [-1]) # force shape
            
            Y = identity([Y_hue Y_sat Y_val])
                        
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