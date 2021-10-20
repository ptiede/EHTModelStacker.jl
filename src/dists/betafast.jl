"""
    BetaFast(α, β)
The beta distribution but faster because I precompute the normalization.

# Notes
Only logpdf is implemented.
"""
struct BetaFast{T} <: Distributions.ContinuousUnivariateDistribution
    α::T
    β::T
    lnorm::T
    function BetaFast(α::T, β::T) where {T}
        lnorm = -logbeta(α, β)
        return new{T}(α, β, lnorm)
    end
end
Distributions.support(::BetaFast{T}) where {T} = zero(T), one(T)

function Distributions.logpdf(d::BetaFast, x::Real)
    return (d.α-1)*log(x) + (d.β-1)*log1p(-x) + d.lnorm
end
