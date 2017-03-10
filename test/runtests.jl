using ColoringNames
using Base.Test
using MLDataUtils

tests = [
    "input.jl",
    "util.jl",
]

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
