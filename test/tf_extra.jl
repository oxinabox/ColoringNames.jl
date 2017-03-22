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




@testset "masks" begin
  mask_sess = Session(Graph())

  m_val = [1, 0, 2, 4]
  M = constant(m_val)

  ### 1D
  maskedM = run(mask_sess, apply_mask(M, get_mask(M)))
  @test maskedM == [1, 2, 4]


  ### 2D
  a_val = (
  [2.  5  8  9  2
   3.  6  4  4  9
   4.  7  3  2  4
   1.  4  2  4  4]
  )
  A=constant(a_val)

  maskedA = run(mask_sess, apply_mask(A, get_mask(M)))

  @test all(maskedA[1,:] .== a_val[1,:])
  @test all(maskedA[2:3,:] .== a_val[3:4,:])

  ### 3D
  b_val = rand(4,10,8)
  B=constant(b_val)


  maskedB = run(mask_sess, apply_mask(B, get_mask(M)))
  @test all(maskedB[1,:,:] .== b_val[1,:,:])
  @test all(maskedB[2:3,:,:] .== b_val[3:4,:,:])
  @test size(maskedB) = (3, 10, 8)

end;
