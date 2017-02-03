function Base.rpad{T}(xs::Vector{T}, n::Integer, p::T=zero(T))
    sizehint!(xs, n)
    while length(xs)<n
        push!(xs, p)
    end
    xs
end

"Creates a matrix where each column is one of the vectors from `xss`"
function rpad_to_matrix{T}(xss::Vector{Vector{T}}, n_rows::T=maximum(length.(xss)), p::T=zero(T))
    n_cols = length(xss) 
    ret = fill(p, (n_rows, n_cols))
    for (cc, xs) in enumerate(xss)
        for (rr, x) in enumerate(xs)
            @inbounds ret[rr, cc] = x
        end
    end
    ret
end

function names_candidates(blk::Expr) :: Vector{Symbol}
    names_in_block = Vector{Symbol}()
    for a in blk.args
        typeof(a) <: Expr || continue
        if a.head == :(=) && isa(a.args[1], Symbol)
            push!(names_in_block, a.args[1])
        else #Recurse, so we captured things in blocks or behind `const`
            append!(names_in_block, names_candidates(a))
        end
    end
    names_in_block
end

"""
Captures the names in this scope into a dictionary, which it returns
usage:
```julia
localnames = @names_from begin
    x = 2
    y = 3
end
@test localnames[:y] == 3
```
Warning: Does not support blocks that contain new scopes within them.
"""
macro names_from(blk::Expr)
    names_in_block = names_candidates(blk)
    namemap = Expr(:call, :Dict, [Expr(:(=>), QuoteNode(nn), nn) for nn in names_in_block]...)
    quote
        begin
            $(esc(blk))
            $(esc(namemap))
        end
    end
end

