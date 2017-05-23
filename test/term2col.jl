using SwiftObjectStores
using ColoringNames
using Base.Test

const cldata = load_monroe_data(;dev_as_train=true, dev_as_test=true)

@testset "TermToColorDistributionNetwork" begin
    mdl = TermToColorDistributionNetwork(cldata.encoding)

    cost_o = train!(mdl, cldata.train.terms_padded, cldata.train.colors; epochs=3)
    @test cost_o[1] > cost_o[end]

    validation_set_results = evaluate(mdl, valid_terms_padded, valid_hsv)
    @test validation_set_results[:perp] < 30
end



@testset "TermToColorDistributionEmpirical" begin
   
    mdl = TermToColorDistributionEmpirical()
   
    train!(mdl, cldata.train.text, cldata.train.colors, splay_stddev=splay_std_dev)
    
    validation_set_results = evaluate(mdl, cldata.test.terms_padded, cldata.test.colors)
    @test validation_set_results[:perp] < 30

    addone_smoothed_mdl = laplace_smooth(mdl, cldata.train.text) 

end

