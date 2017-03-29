using Plots
export plot_colors

pyplot() # Can't trust GR not to mess up colors

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

plot_colors([:red, :green, :blue], [:black, :white, colorant"black"];
            row_names=["a", "b", "c"],
            column_names=["X","Y"])

plot_colors([:red, :green, :blue], [:black, :white, colorant"black"];
            row_names=["a", "b", "c"],
            column_names=["X","Y"])
