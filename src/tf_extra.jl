export trailing_matmul

function Base.Math.atan2{T1,T2}(y::Tensor{T1}, x::Tensor{T2}, ϵ=1.0e-12)
    TensorFlow.with_op_name("ATan2") do
        #Hack to generate correctly typed and sized 0s and 1s
        v0 = zero(T2).*x
        v1 = v0 + one(T2)

        # Add a small number to all zeros, to avoid division by zero:
        x = select(x .== v0, x+convert(T1, ϵ), x)
        y = select(y .== v0, y+convert(T1, ϵ), y)


        Θ = atan(y/x) #This will be kept if x>0
        Θ = Θ + select((x<v0) & (y>v0), π.*v1, v0)
        Θ = Θ - select((x<v0) & (y<v0), π.*v1, v0)
        Θ
    end
end


"""
Does a Matrix multiplication on of the final dimention of a tensor with the Matrix
Equivelent  to in julia `(A,B) -> mapslices(Ā->Ā*B, B, 2:3)` (for 3D A)
"""
function trailing_matmul(A,B)
    A_dims = size(A)
    B_dims = size(B)
    A_flat_dims = reduce_prod(A_dims[1:end-1])

    Af = reshape(A, stack([A_flat_dims, A_dims[end]]))
    ABf = Af*B
    AB = reshape(ABf, concat([A_dims[1:end-1], expand_dims(B_dims[end], 1)],1))
    AB
end
