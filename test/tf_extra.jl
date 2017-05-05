using ColoringNames
using Base.Test
using TensorFlow

@testset "atan2" begin
    x=rand(100)
    y= rand(100)
    @test atan2(y,x) ≈ run(Session(Graph()), atan2(constant(y), constant(x)))
end


@testset "Trailing dimention matrix product" begin
    X=rand(5,7,4)
    Y=rand(4,10);
    XY = mapslices(X̄->X̄*Y, X,2:3)
    @assert size(XY) == (5,7,10)


    sess = Session(Graph())

    @test XY == run(sess, trailing_matmul(constant(X),constant(Y)))
end
