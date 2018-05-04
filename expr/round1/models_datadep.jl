using DataDeps

RegisterDataDep("ColoringNames Models",
"""
These are pretrained models for https://github.com/oxinabox/ColoringNames.jl/
as described in the [preprint](https://arxiv.org/abs/1709.09360)

2017, Lyndon White, Roberto Togneri, Wei Liu, Mohammed Bennamoun; **Learning Distributions of Meant Color**.
It is an ~700Mb download.
""",
https://cloudstor.aarnet.edu.au/plus/s/GVOK7kMHhgsFpQp/download,
post_fetch_method=unpack
)
