
"""Higher Order Function,
return a closure, that can use the rules to tokenize an input.
"""
function morpheme_tokenizer(replacement_rules)
    _cache = Dict() #GOLDPLATE: Type specificity
    function tokenize(input)
        get!(_cache, input) do
            for (pat, sub) in replacement_rules
                input = replace(input, pat, sub)
            end
            split(input)
        end
    end
    tokenize
end

function morpheme_tokenizer(rule_csv_file::AbstractString)
    rules = open(rule_csv_file) do fh
        ObsView(readcsv(fh) , ObsDim.First())
    end
    morpheme_tokenizer(rules)
end;

const rules_path = joinpath(dirname(@__FILE__), "..", "data", "replacement_rules.csv") #Proper way, but does not work with juno
const morpheme_tokenize = morpheme_tokenizer(rules_path)

@memoize demarcate(tokens, starter="<S>", ender="</S>") = [starter; tokens; ender]
