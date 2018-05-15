
"Converts and array 3 values between 0 and 1 for HSV to a colorant"
hsv2colorant(x1, x2, x3) = RGB(HSV(360*x1, x2, x3))

hsv2colorant(x::AbstractVector) = hsv2colorant(x...)
function hsv2colorant(hsvs::AbstractMatrix)
    @assert size(hsvs, 2) == 3
    mapslices(hsv2colorant, hsvs, 2)
end

"""
EG:
plot_colors([:red, :green, :blue], [:black, :white, colorant"black"];
            row_names=["a", "b", "c"],
            column_names=["X","Y"])
"""
function plot_colors(column_colors...; column_names=[], row_names=[])

    xs = Int[]
    ys = Int[]
    texts = []
    for (row_ii, name) in enumerate(row_names)
        push!(xs, 1)
        push!(ys, row_ii)
        push!(texts, name)
    end
    scatter(xs, ys; series_annotations=texts, markersize=0)

    xs = Int[]
    ys = Int[]
    colors = []
    for (column_ii, column) in enumerate(column_colors)
        for (row_ii, color) in enumerate(column)
            push!(xs, column_ii+2)
            push!(ys, row_ii)
            push!(colors, color)
        end
    end
    scatter!(xs, ys; color=colors, shape=:square,#, markersize=100,
        xdiscrete_values=[""; ""; column_names],
        legend=nothing
    )
end


"Plots a histograms of HSV"
function plot_hsv(hp::Vector, sp::Vector, vp::Vector; kwargs...)
    nbins = length(hp)
    @assert nbins == length(sp) == length(vp)
    h_max, s_max, v_max = (indmax.([hp, sp, vp]))/nbins
    @show h_max, s_max, v_max
    h_bar_colors = ColoringNames.hsv2colorant([linspace(0.0,1.0, nbins) s_max*ones(nbins) v_max*ones(nbins)])
    s_bar_colors = ColoringNames.hsv2colorant([h_max*ones(nbins) linspace(0.0,1.0, nbins) v_max*ones(nbins)])
    v_bar_colors = ColoringNames.hsv2colorant([h_max*ones(nbins) s_max*ones(nbins) linspace(0.0,1.0, nbins)])
    #
    bar([hp, sp, vp],
        legend = false,
        layout=(1,3),
        linewidth=0,
        seriescolor=[h_bar_colors s_bar_colors v_bar_colors],
        xlabel=["Hue" "Saturation" "Value"],
        ylabel=["Probability" "" ""]; kwargs...)
end
