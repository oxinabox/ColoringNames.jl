using SwiftObjectStores
using DataStructures
using MLDataPattern
export load_monroe_data, rare_descriptions, ColorDatasets, ColorDataset, extrapolation_dataset

immutable ColorDataset{T,TP,TC}
    texts::T
    terms_padded::TP
    colors::TC
end


immutable ColorDatasets{E, CD<:ColorDataset}
    encoding::E
    train::CD
    dev::CD
    test::CD
end

function load_monroe_data(path=datadep"Munroe Color Corpus"; dev_as_train=false, dev_as_test=true)

        const dev_raw = readdlm(joinpath(path,"dev.csv"), '\t')
        const dev_text = dev_raw[:, 1]
        const dev_hsv, dev_terms_padded, encoding = prepare_data(dev_raw; do_demacate=false)
        dev = ColorDataset(dev_text, dev_terms_padded, dev_hsv)

        if dev_as_train
            train = dev
        else
            const train_raw =  readdlm(joinpath(path,"train.csv"), '\t')
            const train_text = train_raw[:, 1]
            const train_hsv, train_terms_padded, _ = prepare_data(train_raw, encoding; do_demacate=false)
            train = ColorDataset(train_text, train_terms_padded, train_hsv)
        end

        if dev_as_test
            test = dev
        else
            const test_raw = readdlm(joinpath(path,"test.csv"), '\t')
            const test_text = test_raw[:, 1]
            const test_hsv, test_terms_padded, _ = prepare_data(test_raw, encoding; do_demacate=false)
            test = ColorDataset(test_text, test_terms_padded, test_hsv)
        end

        ColorDatasets(encoding, train, dev, test)
end

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
function extrapolation_dataset(base_dataset, restricted_texts)
    restricted = Set(restricted_texts)
    train_inds = find(base_dataset.train.texts) do x
        x ∉  restricted
    end

    dev_inds = find(base_dataset.dev.texts) do x
        x ∈  restricted
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
