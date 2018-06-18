
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
    scatter!(xs, ys; color=colors, shape=:square, markersize=00,
        xdiscrete_values=[""; ""; column_names],
        legend=nothing
    )
end


"Plots a histograms of HSV"
function plot_hsv_hist(hp::Vector, sp::Vector, vp::Vector; kwargs...)
    nbins = length(hp)
    @assert nbins == length(sp) == length(vp)
    midpoints = KernelDensity.kde_range((0,1), nbins)
    h_max, s_max, v_max = peak.([hp, sp, vp])
    @show h_max, s_max, v_max
    h_bar_colors = ColoringNames.hsv2colorant([midpoints s_max*ones(nbins) v_max*ones(nbins)])
    s_bar_colors = ColoringNames.hsv2colorant([h_max*ones(nbins) midpoints v_max*ones(nbins)])
    v_bar_colors = ColoringNames.hsv2colorant([h_max*ones(nbins) s_max*ones(nbins) midpoints])
    #
    bar([hp, sp, vp],
        legend = false,
        layout=(1,3),
        linewidth=1,
        #width=1,
        linecolor = [h_bar_colors s_bar_colors v_bar_colors],
        size=(600,200),
        seriescolor=[h_bar_colors s_bar_colors v_bar_colors],
        xlabel=["Hue" "Saturation" "Value"],
        ylabel=["Probability" "" ""]; kwargs...)
end

function plot_hsv_point(hsvs::Matrix, names; kwargs...)
    colors = mapslices(hsv2colorant, hsvs, 1) |> vec
    @assert length(colors) == length(names)
    xs = 1:length(colors)
    ys = ones(length(colors))
    scatter(xs, ys;
        xlim = (0, length(colors)+1),
        ylim = (0.5, 1.5),
        color = colors,
        markersize=30,
        legend=nothing,
        size=(80*(length(colors)+2),200),
        kwargs...
    )
    scatter!(xs, ifelse.(xs.%2.==1, ys.-0.3, ys.+0.3);
        series_annotations=names,
        markeralpha=0,
    )
end


function plot_query(mdl::AbstractPointEstModel, input_text::Vector;  kwargs...)
    plot_hsv_point(query(mdl, input_text), names; title=input_text, kwargs...)              
end

function plot_query(mdl::AbstractDistEstModel, input_text;  kwargs...)
    plot_hsv_hist(query(mdl, input_text)...; title=["" input_text ""], kwargs...)              
end