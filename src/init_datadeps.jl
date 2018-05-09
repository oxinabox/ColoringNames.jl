function __init__()
    register(DataDep("Munroe Color Corpus",
    """
    This is XKCD color data (https://blog.xkcd.com/2010/05/03/color-survey-results/)
    collected by Randall Munroe, in 2010.
    with the results from all the participants.

    With some filtering and spelling normalistation from
    Brian McMahan and Matthew Stone,
    "A Bayesian Model of Grounded Color Semantics",
    Transactions of the ACL, 2015.
    http://mcmahan.io/lux/

    With some minor data munging into nice shape, by Lyndon White in 2016.

    Use of this data remains the responsibility of the user.
    """,
    "https://cloudstor.aarnet.edu.au/plus/s/dwz6rsdG8tOgBA9/download",
    "00488395712f92d4c02c90672ccc302887926cb42331f20227bc9fd727714c49";
    post_fetch_method = fn -> begin
            DataDeps.unpack(fn)
            mv.(joinpath.("monroe",readdir("monroe")), readdir("monroe"))
            rm("monroe")
        end
    ))
    
    register(DataDep("FastText wiki en",
        """
        Dataset: FastText Word Embeddings for English.
        Author: Bojanowski et. al. (Facebook)
        License: CC-SA 3.0
        Website: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

        300 dimentional FastText word embeddings, trained on Wikipedia
        Citation: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

        Notice: this file is ~ 10GB
        """,
        "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip",
        "f4d87723baad28804f89c0ecf74fd0f52ac2ae194c270cb4c89b0a84f0bcf53b";
        post_fetch_method=DataDeps.unpack
    ));

    register(DataDep("FastText news en",
        """
        Dataset: FastText Word Embeddings for English.
        Author: Bojanowski et. al. (Facebook)
        License: CC-SA 3.0
        Website: https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

        300 dimentional FastText word embeddings, trained on Wikipedia
        Citation: P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

        Notice: this file is ~ 0.6GB
        """,
        "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip",
        "bdeb85f44892c505953e3654183e9cb0d792ee51be0992460593e27198d746f8",
        post_fetch_method=DataDeps.unpack
    ));
    
    
    
    register(DataDep("word2vec 300d",
    """
    Pretrained Word2Vec Word emeddings
    Website: https://code.google.com/archive/p/word2vec/
    Author: Mikolov et al.
    Year: 2013

    Pre-trained vectors trained on part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases.

    Paper:
        Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
    """,
    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz",
    "21c05ae916a67a4da59b1d006903355cced7de7da1e42bff9f0504198c748da8";
    post_fetch_method=DataDeps.unpack
    ))

end
