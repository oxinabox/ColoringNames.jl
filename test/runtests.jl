using ColoringNames
using Base.Test



tests = [
    "prob.jl",
    "prob2.jl"
    "input.jl",
    "util.jl",
    "tf_extra.jl",
    "evalutation.jl",
    "embeddings.jl",
    #Long Run Tests Tests
    "term2col.jl"
]

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
