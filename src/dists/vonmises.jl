"""
    VonMisesWrap(μ, σ)
This is a von Mises distribution but without the branch cut present in Distributions.jl
This prevent some funny buisness when θ isn't in [-π,π].

# Notes
Only logpdf is implemented! So don't call rand.
"""
struct VonMisesWrap{T,S} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::S
    I0κx::S
end

function VonMisesWrap(μ, σ)
    VonMisesWrap(μ, σ, besselix(zero(typeof(σ)), 1/σ^2))
end

@inline function Distributions.logpdf(dist::VonMisesWrap, x::Real)
    μ,σ = dist.μ, dist.σ
    dθ = (cos(x-μ)-1)/σ^2
    return dθ - log(dist.I0κx) - log2π
end

Base.minimum(::VonMisesWrap) = -Inf
Base.maximum(::VonMisesWrap) = Inf
