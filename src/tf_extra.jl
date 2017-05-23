
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

