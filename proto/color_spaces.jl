using SwiftObjectStores
using ColoringNames
using Distributions
using MLDataUtils
using Iterators
using MLLabelUtils
using StaticArrays
using Juno
using StatsBase

using ColorTypes
using Colors

const od =(MLDataUtils.ObsDim.First(), MLDataUtils.ObsDim.Last())
const serv=SwiftService()

const valid_raw = get_file(fh->readdlm(fh,'\t'), serv, "color", "monroe/dev.csv")
const valid_hsv, valid_terms_padded, encoding = prepare_data(valid_raw)

##################################


colors = ColoringNames.hsv2colorant(valid_hsv)[:]

XYZ_cols = convert.(Scalar(xyY), colors)

using Plots
plotlyjs()
gr()

sub = shuffle(XYZ_cols)[1:1000]
scatter3d((c->c.x).(sub), (c->c.y).(sub), (c->c.Y).(sub),
    color=sub,
    legend=false)

sub_mat = getfield.([sub sub sub],[:x :y :z])

sub_mat[:,1]+sub_mat[:,1]


Plots.rgba_string.(convert.(Scalar(RGB), sub))

sub











#EOF
