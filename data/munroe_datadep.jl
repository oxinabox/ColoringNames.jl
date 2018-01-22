using DataDeps

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
post_fetch_method=unpack
)
