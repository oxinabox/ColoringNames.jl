using ColoringNames

runname = "sowe1"
logdir = mkpath(joinpath(@__DIR__, "logs", runname))


word_vecs, vocab, enc = load_word2vec_embeddings()
const cldata = load_monroe_data(
    dev_as_train=false,
    dev_as_test=true,
    encoding=enc)




println("initialising $runname network")
const mdl = TermToColorDistributionSOWE(word_vecs)

println("estimating raw distribution")
text, terms, find_distribution(colors, nbins)

println("training $runname network")
train!(mdl,
        cldata.train.terms_padded,
        cldata.train.colors,
        log_path;
        batch_size = batch_size,
        epochs=epochs
        )
