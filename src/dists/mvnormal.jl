"""
    MvNormalFast
This is a specialized MvNormal that assumes a **diagonal** covariance. This is substantially
faster than the usual Distributions.jl version that for some reason has large overhead.

# Notes
Only logpdf is defined so no calling rand!
"""
struct MvNormalFast{T<:AbstractVector, N} <: Distributions.AbstractMvNormal
    μ::T
    Σ::T
    lnorm::N
    function MvNormalFast(μ::T, Σ::T) where {T}
        lnorm = -0.5*log(prod(Σ)) - length(μ)/2*log(2π)
        return new{typeof(μ), typeof(lnorm)}(μ, Σ, lnorm)
    end
end

@inline function Distributions.logpdf(d::MvNormalFast, x::AbstractVector)
    μ,Σ = d.μ, d.Σ
    acc = zero(eltype(x))
    @turbo for i in eachindex(x)
        acc += -(x[i]-μ[i])^2/Σ[i]/2
    end
    return acc + d.lnorm
end

@inline Distributions.pdf(d::MvNormalFast, x) = exp(logpdf(d, x))


"""
    MvNormal2D
A specialized MvNormal just for 2D since there are nice parameterizations for this.
You basically get loop unrolling.
"""
struct MvNormal2D{T,C}
    μ::T
    Σ::C
end

function Distributions.pdf(d::MvNormal2D, x)
    μ,Σ = d.μ, d.Σ
    σx = sqrt(Σ[1,1])
    σy = sqrt(Σ[2,2])
    ρ = Σ[2,1]/(σx*σy)
    ρinv = 1-ρ^2
    fx = ((x[1]-μ[1])/σx)^2
    fy = ((x[2]-μ[2])/σy)^2
    fxy = -2*ρ*((x[1]-μ[1])/σx)*((x[2]-μ[2])/σy)

    return 1/(2π*σx*σy*sqrt(ρinv))*
            exp(-1/(2*ρinv)*(fx+fy+fxy))

end
