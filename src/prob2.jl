function find_distributions(data::ColorDataset, nbins)

    map(groupby(last, enumerate(data.texts))) do grp
        ind_start, text = first(grp)
        ind_end  = first(last(grp))

        inds = ind_start:ind_end # faster to slice with ranges
        terms = data.terms_padded[:, first(inds)]
        colors = data.colors[inds, :]

        text, terms, find_distribution(colors, nbins)
    end
end

function find_distribution(colors::AbstractMatrix, nbins)
    map(obsview(colors)) do ch
        find_distribution(ch, nbins)
    end
end

function find_distribution(channel::AbstractVector, nbins)
    # PREMOPT: THis could be much faster if we just looped through each edge
    @assert(all(0 .<= channel .<= 1))

    starts = 0 : 1/nbins : 1-1/nbins
    ends = starts .+ 1/nbins
    hits = starts .<= channel' .<= ends
    normed_hits = mapslices(hits,1) do col
        # Split anything that was on the boundry into 0.5 in each side
        col/sum(col)
    end
    mean(normed_hits, 2) |> vec
end








#
