using DataDeps
include("init_datadeps.jl")()



#Loads googles word2vec_embeddings
function load_word2vec_embeddings(embedding_file, max_stored_vocab_size = 1_000_000, keep_words=Set())
    #If there are anyt words in keep_words, then only those are kept, otherwise all are kept

    #Note: I know there actually <10^6 words in the vocab, when phrases are exluded, so lock the vocab size to this to save 70%RAM
    #Words are loosely organised by commonness,  AFAICT
    fh = open(embedding_file,"r")
    vocab_size, vector_size = parse.(Int64, split(readline(fh)))
    max_stored_vocab_size = min(max_stored_vocab_size, vocab_size) #if using a small vocab then there is a chance you might be willing ot store more words than it has


    indexed_words = Array{String}(max_stored_vocab_size)
    word_indexes = Dict{String,Int64}()
    LL = Array{Float32}(vector_size, max_stored_vocab_size)

    index = 1
    for _ in 1:vocab_size
        word = readuntil(fh,' ') |> strip #Technically this is 'ISO-8859-1' may have to deal with encoding issues
        vector = read(fh, Float32,vector_size )

        if !contains(word, "_") && (length(keep_words)==0 || word in keep_words ) #If it isn't a phrase
            LL[:,index]=vector./norm(vector)
            indexed_words[index] = word
            word_indexes[word] = index

            index+=1
            if index>max_stored_vocab_size
                warn("Max Vocab size exceeded. More words are available if you want.")
                break
            end
        end

    end
    close(fh)

    LL = LL[:,1:index-1] #throw away unused columns
    indexed_words = indexed_words[1:index-1] #throw away unused columns
    LL, word_indexes, indexed_words
end


word_vecs, word_indexes, vocab = load_word2vec_embeddings(
    datadep"word2vec 300d/GoogleNews-vectors-negative300.bin";
    )
vocab, word_vecs

haskey(word_indexes, "purplish")

wv(word) = word_vecs[:,word_indexes[word]]

mean.(abs.(wv("purplish") - (wv("purple") + wv("ish"))))
sumabs(wv("purple") - (wv("red") + wv("blue")))
sumabs(wv("orange") - (wv("red") + wv("yellow")))
sumabs(wv("orange"))









#
