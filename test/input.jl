using ColoringNames
using Base.Test

@testset "Tokenize" begin
    rules = [
        ("blueish","blue ish"),
        ("bluish","blue ish"),
        ("-", " - "),
        ("purplish","purple ish")]

    tokenizer = morpheme_tokenizer(rules)

    @test tokenizer("green") == ["green"]
    @test tokenizer("a b") == ["a", "b"]
    @test tokenizer("bluish red") == ["blue", "ish","red"]
    @test tokenizer("red-green") == ["red", "-","green"]


end

@test demarcate(["a", "b", "c"]) == ["<S>", "a", "b", "c", "</S>"]


@testset "Rare Desc" begin
    eg = split("""a 1
    b 2
    b 3
    b 3
    a 3
    c 3
    c 3
    c 3
    d 9""", "\n")
    @test rare_desciptions(eg, 5, 0)|> Set == ["a 1", "a 3", "d 9", "b 2", "b 3"] |> Set
    @test rare_desciptions(eg, 30, 1) |> Set == ["a 3", "b 3"] |> Set
end