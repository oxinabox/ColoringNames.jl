
type ProgressiveSubSamplerState{T,S}
    total::Int
    counts::Dict{T, Int}
    pending::T
    source_state::S
end

immutable ProgressiveSubSampler{A}
    source::A
end

Base.eltype()

srand(1)
src = rand([1,2,2,3,3,3], 10_000)
