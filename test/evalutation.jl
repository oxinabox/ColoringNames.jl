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

@testset "bin expected value" begin
    @test bin_expected_value([1,2,3,4], 4) == [1/8, 3/8, 5/8, 7/8 ]
    @test bin_expected_value([1,2,3], 3) == [1/6, 1/2, 5/6 ]
end

@testset "Peak" begin
    
    @test peak([0.5, 2.5, 0.3]) == 1/2
    @test peak([0.5 0.2 0.3; 0.1 0.1 0.8]) ≈ [1/6, 5/6]


    three181 = [0.1 0.8 0.1; 0.1 0.8 0.7; 0.1 0.2 0.1]
    @test peak(three181) == reshape([0.5; 0.5; 0.5], (3,1))
    three127 = [0.1 0.2 0.7; 0.1 0.2 0.7; 0.1 0.2 0.7]
    @test peak(three127) == reshape([5/6; 5/6; 5/6], (3,1))

end
