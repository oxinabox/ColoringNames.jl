using DataDeps
include("init_datadeps.jl")()


# From Word2Vec.jl
# Copying here as that is not being maintained
function wordvectors(filename::AbstractString)
    open(filename) do f
        header = strip(readline(f))
        vocab_size,vector_size = map(x -> parse(Int, x), split(header, ' '))
        vocab = Vector{AbstractString}(vocab_size)
        vectors = Array{Float32}(vector_size, vocab_size)
        binary_length = sizeof(Float32) * vector_size
        for i in 1:vocab_size
            vocab[i] = strip(readuntil(f, ' '))
            vector = read(f, Float32, vector_size)
            vec_norm = norm(vector)
            vectors[:, i] = vector./vec_norm  # unit vector
            read(f, UInt8) # new line
        end
        return vocab, vectors
    end
end

vocab, word_vecs = wordvectors(
    datadep"word2vec 300d/GoogleNews-vectors-negative300.bin";
    )
vocab, word_vecs
