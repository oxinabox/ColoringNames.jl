using ColoringNames
using Base.Test

function test_ispmf(res, tol=1e-15)
    @test isapprox(sum(res), 1.0; atol=tol)
    @test all(res.>=0.0)
end

@testset "GaussianHot" begin
    res=gaussianhot(0.0, 10)
    @test all(res[1].>res[2:end])
    test_ispmf(res)
    @test length(res)==10

    res=gaussianhot(1.0, 10)
    @test all(res[10].>res[1:9])
    test_ispmf(res)
    @test length(res)==10

    res=gaussianhot(0.51, 10)
    @test all(res[6].>res[1:5])
    @test all(res[6].>res[7:end])
    test_ispmf(res)
    @test length(res)==10

    res=gaussianhot(0.5, 10) #right on the boarder
    @test res[5] ≈ res[6]
    @test all(res[6].>res[1:4])
    @test all(res[6].>res[7:end])
    test_ispmf(res)
    @test length(res)==10
end


@testset "vonmiseshot" begin

    res=vonmiseshot(0.51, 10)
    @test all(res[6].>res[1:5])
    @test all(res[6].>res[7:end])
    @test all(res[end-1].>res[end])
    test_ispmf(res, 0.2)
    @test length(res)==10

    res=vonmiseshot(-pi, 10, -pi, pi)
    @test res[1] ≈ res[end]
    @test res[2] ≈ res[end-1]
    @test res[2]<res[1]

    @test vonmiseshot(0.0, 10) ≈  vonmiseshot(-pi, 10, -pi, pi)
    @test vonmiseshot(0.0, 10) ≈ vonmiseshot(1.0, 10)

    for val in 0:0.01:1
        res=vonmiseshot(val, 256)
        test_ispmf(res)
    end
end


@testset "gaussianwraparoundhot" begin

    res=gaussianwraparoundhot(0.51, 10)
    @test all(res[6].>res[1:5])
    @test all(res[6].>res[7:end])
    @test all(res[end-1].>res[end])
    test_ispmf(res, 0.2)
    @test length(res)==10

    res=gaussianwraparoundhot(-pi, 10, -pi, pi)
    @test res[1] ≈ res[end]
    @test res[2] ≈ res[end-1]
    @test res[2]<res[1]

    @test gaussianwraparoundhot(0.0, 10) ≈  gaussianwraparoundhot(-pi, 10, -pi, pi)
    @test gaussianwraparoundhot(0.0, 10) ≈ gaussianwraparoundhot(1.0, 10)

    for val in 0:0.01:1
        res=gaussianwraparoundhot(val, 256)
        test_ispmf(res)
    end
end

#=
using Plots
gr()
bar(gaussianwraparoundhot(0.51, 10))

bar(gaussianwraparoundhot(-pi, 10, -pi, pi))
=#
