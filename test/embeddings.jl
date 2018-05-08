using ColoringNames
using DataDeps
using Base.Test

word_vecs, vocab, enc = load_word2vec_embeddings()
vocab, word_vecs

@test length(vocab) == size(word_vecs, 2) == nlabel(enc)

@test label2ind("purplish", enc)== findfirst(vocab .== "purplish")
@test label2ind("purple", enc)== findfirst(vocab .== "purple")
@test label2ind("ish", enc)== findfirst(vocab .== "ish")
@test label2ind("blue", enc)== findfirst(vocab .== "blue")






#
