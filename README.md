# ColoringNames

This software is written in the Julia programming language.
All requirements are given in the REQUIRE file.


## Directories:


 - `data` contains code for downloading the data, and the set of rules for tokenising it
 - `expr` contains code for running the experiments, and analysing the results
    - Of particular interested is PaperGen.ipynb which generates all the tables and figures used in the paper.
    - This code is a bit messy honestly some files are out of date or for older versions.
    - Most of the expriments will not run without modifications as the training/testing data is not hosted in publicly accessible locations
    - You can rebuild this data following the setup in `data`
 - `src` contains the the code for the models discussed
 - `test` contains tests asserting the code is working as intended
 
## Note on naming:

 - The CDEST model is in code called the `Term2ColorDistributionNetwork`
 - the Baseline model is in code called the `laplace_smoothed` `TermToColorDistributionEmpirical`, it is also often refered to as `noml` model

