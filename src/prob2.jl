find_distributions(data::ColorDataset, nbins, smooth::Bool) = find_distributions(data, nbins, Val{smooth}())

function find_distributions(data::ColorDataset, nbins, smooth::Val, T=Float32)
    
    grps = collect(groupby(last, enumerate(data.texts)))
    len = length(grps)
    nsteps = size(data.terms_padded,1)

    texts = Vector{String}(len)
    terms = Matrix{Int}(nsteps, len)
    hs = Matrix{T}(nbins, len)
    ss = Matrix{T}(nbins, len)
    vs = Matrix{T}(nbins, len)

    for (ii, grp) in enumerate(grps)
        
        ind_start, texts[ii] = first(grp)
        ind_end  = first(last(grp))
        
        inds = ind_start:ind_end # faster to slice with ranges
        terms[:, ii] = data.terms_padded[:, first(inds)]
        
        colors = @view data.colors[inds, :]
        hs[:,ii], ss[:,ii], vs[:,ii] = find_distribution(colors, nbins, smooth)
    end
    texts, terms, (hs, ss, vs)
end

function find_distribution(colors::AbstractMatrix, nbins, smooth::Val{false})
    hs = do_not_smooth(@view(colors[:,1]), nbins) |> first
    ss = do_not_smooth(@view(colors[:,2]), nbins) |> first
    vs = do_not_smooth(@view(colors[:,3]), nbins) |> first
    hs, ss, vs
end

function find_distribution(colors::AbstractMatrix, nbins, smooth::Val{true})
    hs = wraparound_kde_smooth((@view colors[:,1]), nbins) |> first
    ss = truncated_kde_smooth((@view colors[:,2]), nbins) |> first
    vs = truncated_kde_smooth((@view colors[:,3]), nbins) |> first
    hs, vs, ss
end


## Smoothers 

function do_not_smooth(data, npoints)
    midpoints = KernelDensity.kde_range((0,1), npoints)
    dist = KernelDensity.tabulate(data, midpoints)
    dist.density./=sum(dist.density)
    dist.density, dist.x
end


function truncated_kde_smooth(data, npoints, kde_fun = kde_lscv)
    fake_boundry = (-0.5, 1.5)
    fake_npoints = 2npoints

    dist = kde_fun(data, npoints=fake_npoints, boundary=fake_boundry)
    density = dist.density
    
    density./=sum(density)
    inside = density[npoints÷2+1: end - npoints÷2]
        
    @assert length(inside)==npoints
    excess_mass = 1-sum(inside)
    @assert 0<=excess_mass<=0.5 
    inside./= 1-excess_mass
    @assert sum(inside)≈ 1
    inside, dist.x[npoints÷2+1: end - npoints÷2]
end

function wraparound_kde_smooth(data, npoints, kde_fun = kde_lscv)
    # Because of the periodic nature of FFT used to implement kde
    # It is actually wrap around by default
    # If you specify tight boundries
    boundry = (0, 1)
    dist = kde_fun(data, npoints=npoints, boundary=boundry)
    dist.density./=sum(dist.density)
    
    dist.density, dist.x
end


#
