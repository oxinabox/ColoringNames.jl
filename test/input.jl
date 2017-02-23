using ColoringNames
using Base.Test

@testset "Tokenize" begin
    rules = ObsView(readcsv(IOBuffer(
            """blueish,blue ish
            bluish,blue ish
            -, -
            purplish,purple ish""")) , ObsDim.First())

    tokenizer = morpheme_tokenizer(rules)

    @test tokenizer("a b") == ["a", "b"]
    @test tokenizer("bluish red") == ["blue", "ish","red"]
    @test tokenizer("red-green") == ["red", "-","green"]
end

@test demarcate(["a", "b", "c"]) == ["<S>", "a", "b", "c", "</S>"]
