immutable ColorDataset{T,TP,TC}
    texts::T
    terms_padded::TP
    colors::TC
end


immutable ColorDatasets{E, CD1<:ColorDataset, CD2<:ColorDataset, CD3<:ColorDataset}
    encoding::E
    train::CD1
    dev::CD2
    test::CD3
end



function load_munroe_data(path=datadep"Munroe Color Corpus"; dev_as_train=false, dev_as_test=true, encoding_=nothing)
    encoding = encoding_ #weird bug seems to not like encoding as a kwarg
    const dev_raw = readdlm(joinpath(path,"dev.csv"), '\t')
    const dev_text = dev_raw[:, 1]
    const dev_hsv, dev_terms_padded, encoding = prepare_data(dev_raw, encoding)
    dev = ColorDataset(dev_text, dev_terms_padded, dev_hsv)

    if dev_as_train
        train = dev
    else
        const train_raw =  readdlm(joinpath(path,"train.csv"), '\t')
        const train_text = train_raw[:, 1]
        const train_hsv, train_terms_padded, _ = prepare_data(train_raw, encoding)
        train = ColorDataset(train_text, train_terms_padded, train_hsv)
    end

    if dev_as_test
        test = dev
    else
        const test_raw = readdlm(joinpath(path,"test.csv"), '\t')
        const test_text = test_raw[:, 1]
        const test_hsv, test_terms_padded, _ = prepare_data(test_raw, encoding)
        test = ColorDataset(test_text, test_terms_padded, test_hsv)
    end

    ColorDatasets(encoding, train, dev, test)
end

##########################################################################################
# Extrapolation Dataset


"""
Finds rare color descriptions.

- `min_remaining_tokens` the minimum number of instances of every token that must remain in the training set
- this only counts tokens that would be up for deletion due to their being part of a term that was considered for rarity
"""
function rare_descriptions(train_descs, n_rarest=100, min_distinct_remaining_tokens_usages=8;  tokenize = morpheme_tokenize)
    desc_freqs = sort(labelfreq(train_descs), byvalue=true)

    token_unique_usages = counter(AbstractString[])
    for desc in keys(desc_freqs)
        for tok in tokenize(desc)
            push!(token_unique_usages, tok)
        end
    end


    rares = String[]

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


"""
Defined a new dataset for purposes of evaluating a methods ability to extrapolate.

Given a Dataset and a list of `texts`, removes all instances of those texts from the train set,
and restricts the dev and test sets to *only* contain those `texts`
"""
function extrapolation_dataset(base_dataset, restricted_texts=rare_descriptions(base_dataset.train.texts))
    restricted = Set(restricted_texts)
    train_inds = find(base_dataset.train.texts) do x
        x ∉  restricted
    end

    dev_inds = find(base_dataset.dev.texts) do x
        x ∉  restricted
    end

    test_inds = find(base_dataset.test.texts) do x
        x ∈  restricted
    end
    od = (ObsDim.Last(), ObsDim.Last(), ObsDim.First())

    ColorDatasets(base_dataset.encoding,
        ColorDataset(datasubset((base_dataset.train.texts, base_dataset.train.terms_padded, base_dataset.train.colors), train_inds, od)...),
        ColorDataset(datasubset((base_dataset.dev.texts, base_dataset.dev.terms_padded, base_dataset.dev.colors), dev_inds, od)...),
        ColorDataset(datasubset((base_dataset.test.texts, base_dataset.test.terms_padded, base_dataset.test.colors), test_inds, od)...)
    )


end

########################################################################################################################
# Order Relevances

function order_relevant_name_pairs(cldataset::ColorDataset)
    inds = findfirst.([cldataset.texts], unique(cldataset.texts)); #Index for elements with unique texts
    terms = mapslices(x->[x], cldataset.terms_padded[:,  inds], 1) |> vec; # the terms-vect encodings for those elements

    sort_on = Tuple.(sort.(terms)) # convert terms vector to sorted tuples
    # so when we sort it later, [2,0,1] will be next to [1,2,0]

    sorted = sort(collect(zip(sort_on, inds)), by=first) # sort it based on sorted_on, pairing it with the the inds 
    grouped = groupby(first, sorted) # group it based on sorted_on


    order_relevant_ind_grps = [last.(grp) for grp in grouped 
                        if length(grp) > 1 && sum(grp[1][1] .!==0) != 1] 
                        #if sum(grp[1][1] .!==0) == 1 then it is asingle word, so prob a typo

    # check to make sure everything is actually unique
    for inds in order_relevant_ind_grps
        ii1, ii2 = inds
        @assert cldataset.terms_padded[:,ii1] !== cldataset.terms_padded[:,ii2]
    end

    getindex.([cldataset.texts], order_relevant_ind_grps)
end

            
"""
    order_relevant_dataset(base_dataset)
            
            Generates a new dataset with the same test dataset, but with a 
"""
function order_relevant_dataset(base_dataset)
    keep_texts = reduce(vcat,Vector{String}, order_relevant_name_pairs(base_dataset.dev))

    dev_inds = find(base_dataset.dev.texts) do x
        x ∈  keep_texts
    end

    test_inds = find(base_dataset.test.texts) do x
        x ∈  keep_texts
    end
    od = (ObsDim.Last(), ObsDim.Last(), ObsDim.First())

    ColorDatasets(base_dataset.encoding,
        base_dataset.train,
        ColorDataset(datasubset((base_dataset.dev.texts, base_dataset.dev.terms_padded, base_dataset.dev.colors), dev_inds, od)...),
        ColorDataset(datasubset((base_dataset.test.texts, base_dataset.test.terms_padded, base_dataset.test.colors), test_inds, od)...)
    )
end




#################################################



function load_color_nameset(tokenize = morpheme_tokenize)
    many_names = reduce(union!, Set{String}(), tokenize.(collect(keys(NamedColors.ALL_COLORS))))
    union!(many_names, ["caucasian", "darker", "pretty", "nude", "skin", "spam", "yuck", "hiccup", "horrible", "colour", "flourescent", "tone", "flesh"])
    union!(many_names, ["newspaper", "goverment", "friendly", "danger", "safety"])
    union!(many_names, map(lowercase, many_names))
    many_names
end
