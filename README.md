# ColoringNames

[![Build Status](https://travis-ci.org/oxinabox/ColoringNames.jl.svg?branch=master)](https://travis-ci.org/oxinabox/ColoringNames.jl)

[![Coverage Status](https://coveralls.io/repos/oxinabox/ColoringNames.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/oxinabox/ColoringNames.jl?branch=master)

[![codecov.io](http://codecov.io/github/oxinabox/ColoringNames.jl/coverage.svg?branch=master)](http://codecov.io/github/oxinabox/ColoringNames.jl?branch=master)

# ColoringNames

This software is written in the Julia programming language.
All requirements are given in the REQUIRE file.


## Directories:

 - `data` contains code for downloading the data, and the set of rules for tokenising it
 - `expr` contains code for running the experiments, and analysing the results
    - Of particular interested is PaperGen.ipynb which generates all the tables and figures used in the paper.
 - `src` contains the the code for the models discussed
 - `test` contains theses asserting the code is working as intended
 
## Note on naming:

 - The CDEST model is in code called the `Term2ColorDistributionNetwork`
 - the Baseline model is in code called the `laplace_smoothed` `TermToColorDistributionEmpirical`, it is also often refered to as `noml` model


## On Blinding and  Open-source

This code has been made available on Github in the past, but for purposes of blind review it has been made private.
However, the author's identities can never the less be trivially broken by looking at the contributors to the packages this software requires.
Many open-source contributions were made as part of this work. 

