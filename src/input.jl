using SwiftObjectStores
using DataStructures
export load_monroe_data, rare_desciptions


macro load_monroe_data(valid_as_train=false)
    quote
        const serv=SwiftService()
        
        const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
        const valid_text = valid_raw[:, 1]
        const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw; do_demacate=false)
        
        if $(valid_as_train)
            train_raw = valid_raw
            train_text = valid_text
            train_hsv = valid_hsv
            train_terms_padded = valid_terms_padded
        else
            const train_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/train.csv")
            const train_text = train_raw[:, 1]
            const train_hsv, train_terms_padded, encoding = prepare_data(train_raw, encoding; do_demacate=false)
            
        end
    end |> esc

end


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


"""
Finds rare color descriptions.

- `min_remaining_tokens` the minimum number of instances of every token that must remain in the training set
- this only counts tokens that would be up for deletion due to their being part of a term that was considered for rarity
"""
function rare_desciptions(train_descs, n_rarest, min_distinct_remaining_tokens_usages=8;  tokenize = morpheme_tokenize)
    desc_freqs = sort(labelfreq(train_descs), byvalue=true)
    
    token_unique_usages = counter(AbstractString[])
    for desc in keys(desc_freqs)
        for tok in tokenize(desc)
            push!(token_unique_usages, tok)
        end
    end
    
    
    rares = eltype(train_descs)[]
       
    for desc in keys(desc_freqs)
        if any(token_unique_usages[tok]<=min_distinct_remaining_tokens_usages for tok in tokenize(desc))
            continue
        end
        push!(rares, desc)
        for tok in tokenize(desc)
            push!(token_unique_usages, tok, -1)
        end
                
        length(rares)>=n_rarest && break #GOLDPLATE: I really like the idea of doing this lazy and then just letting the consumer truncate it at will
    end
    rares
end
    
    