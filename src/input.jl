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
            string.(split(input)) # Convert to plain strings, so they do not hold the original file as references
        end
    end
    tokenize
end

function morpheme_tokenizer(rule_csv_file::AbstractString)
    rules = open(rule_csv_file) do fh
        ObsView(readcsv(fh, comments=true), ObsDim.First())
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
function prepare_data(raw, encoding_=nothing; do_demacate=true, tokenizer=morpheme_tokenize)
    labels = convert(Vector{String}, raw[:,1]);
    labels_padded, encoding = prepare_labels(labels, encoding_; do_demacate=do_demacate, tokenizer=tokenizer)

    hsv_data = convert(Matrix{Float32}, raw[:,2:end]);
    hsv_data, labels_padded, encoding
end

function prepare_labels(labels, encoding_=nothing; do_demacate=true, tokenizer=morpheme_tokenize)

    tokenized_labels = tokenizer.(labels)
    if do_demacate
        tokenized_labels = demarcate.(tokenized_labels)
    end

    local encoding
    if encoding_===nothing
        all_tokens = reduce(union, tokenized_labels)
        encoding = labelenc(all_tokens)
    else
        encoding = encoding_
    end

    label_inds = map(toks->label2ind.(toks, Scalar(encoding)), tokenized_labels)

    rpad_to_matrix(label_inds), encoding
end













#
