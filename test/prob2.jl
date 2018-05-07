using ColoringNames
using Base.Test

datasets = load_monroe_data()
data= datasets.train
dists = ColoringNames.find_distributions(datasets.dev, 256)
for dist in dists
    @test dist[1] |> typeof <: AbstractString
    @test dist[2] |> typeof <: AbstractVector{<:Integer}
    channels = dist[3]
    @test length(channels)==3
    for channel in channels
        @test channel |> typeof <: AbstractVector{<:Real}
        @test channel |> length == 256
        @test all(channel .>= 0)
        @test sum(channel) ≈ 1
    end
end
