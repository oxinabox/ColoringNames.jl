using ColoringNames
using Base.Test
using TensorFlow

@testset "atan2" begin
    x=rand(100)
    y= rand(100)
    @test atan2(y,x) ≈ run(Session(Graph()), atan2(constant(y), constant(x)))
end


