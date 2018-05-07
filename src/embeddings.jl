using Word2Vec
using DataDeps
include("init_datadeps.jl")()
wvs = wordvectors(
    datadep"word2vec 300d/GoogleNews-vectors-negative300.bin";
     kind=:binary)
wvs
