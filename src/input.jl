
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



"""
Encodes and packs data
Returns the packed data and the encoder.
"""
function prepare_data(raw, encoding_=nothing, tokenize=morpheme_tokenize)
    labels = convert(Vector{String}, raw[:,1]);
    hsv_data = convert(Matrix{Float64}, raw[:,2:end]);
    tokenized_labels = demarcate.(tokenize.(labels))
    
    local encoding
    if encoding_===nothing
        all_tokens = reduce(union, tokenized_labels)
        encoding = labelenc(all_tokens)
    else
        encoding = encoding_
    end

    label_inds = map(toks->label2ind.(toks, Scalar(encoding)), tokenized_labels)
    rpad_to_matrix(label_inds), hsv_data, encoding
end
