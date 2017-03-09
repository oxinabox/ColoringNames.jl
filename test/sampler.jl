using InterfaceTesting
using ColoringNames
using StatsBase
using Base.Test
using MLDataUtils

test_iterator_interface(ProgressiveUnderSampler{Int})

srand(1)

@testset "Oversample, basic" begin
    n_src = 2000
    src = rand([1,2,2,3,3,3, 4,4,4,4], 2000)
    oversampled = oversample(src)
    @test all(counts(oversampled).==counts(oversampled)[1])
    @test all( x ∈ oversampled for x in unique(src))
end



@testset "Oversample, MLDataUtils" begin
    n_src = 20
    lbs = rand([1,2,2,3,3,3, 4,4,4,4], n_src)
    data = rand(n_src, 5) #50 features

    od = MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.First()
    oversample((lbs, data); obsdim=od)
        @show a
    end
end
