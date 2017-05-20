using SwiftObjectStores
using ColoringNames
using Base.Test

ColoringNames.@load_monroe_data(true)

@testset "TermToColorDistributionNetwork" begin
    mdl = TermToColorDistributionNetwork(encoding)

    cost_o = train!(mdl, train_terms_padded, train_hsv)
    @test cost_o[1] > cost_o[end]

    validation_set_results = evaluate(mdl, valid_terms_padded, valid_hsv)
    @test validation_set_results[:perp] < 30
end



@testset "TermToColorDistributionEmpirical" begin
   
    mdl = TermToColorDistributionEmpirical()
   
    train!(mdl, train_text, train_hsv, splay_stddev=splay_std_dev)
    
    validation_set_results = evaluate(mdl, valid_terms_padded, valid_hsv)
    @test validation_set_results[:perp] < 30

    addone_smoothed_mdl = laplace_smooth(mdlm 

end

