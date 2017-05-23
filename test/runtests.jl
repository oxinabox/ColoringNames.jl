using ColoringNames
using Base.Test



tests = [
    "prob.jl",
    "input.jl",
    "util.jl",
    "tf_extra.jl",
    "evalutation.jl",
    
    #Long Run Tests Tests
    "term2col.jl"
]

for filename in tests
    name = first(splitext(filename))
    @testset "$name" begin
        include(filename)
    end
end
