function __init__()
    RegisterDataDep("Munroe Color Corpus",
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
    post_fetch_method=unpack
    )

    RegisterDataDep("word2vec 300d",
    """
    Pretrained Word2Vec Word emeddings
    Website: https://code.google.com/archive/p/word2vec/
    Author: Mikolov et al.
    Year: 2013

    Pre-trained vectors trained on part of Google News dataset (about 100 billion words). The model contains 300-dimensional vectors for 3 million words and phrases.

    Paper:
        Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
    """,
    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz";
    post_fetch_method=unpack
    )

end
