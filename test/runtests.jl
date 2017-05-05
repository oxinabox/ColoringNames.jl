using ColoringNames
using Base.Test
using MLDataUtils

tests = [
    "prob.jl",
    "input.jl",
    "util.jl",
    "tf_extra.jl",
    "evalutation.jl"

]

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
