using ColoringNames
using Base.Test

@testset "descretized_perplexity" begin
    two127 = [0.1 0.2 0.7; 0.1 0.2 0.7;]
    @test descretized_perplexity([3,3]/3, two127) <
        descretized_perplexity([2,3]/3, two127) <
        descretized_perplexity([1,3]/3, two127) ==
        descretized_perplexity([3,1]/3, two127) <
        descretized_perplexity([1,1]/3, two127)

    @test descretized_perplexity([0.2,0.2, 0.2, 0.2], ones(4, 10)./10) ≈ 10
    @test descretized_perplexity([0.1,0.7, 0.8, 0.1], ones(4, 10)./10) ≈ 10
end
