export CauchyFast

"""
    CauchyFast(μ,σ)
This creates a Cauchy distribution that is an order of magnitude faster than the
Distributions.jl implementation since is computes the Cauchyization value at
creation since this will be constant. This is a limited distribution though, and
only have logpdf, logcdf, and support implemented.
"""
struct CauchyFast{T} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    lnorm::T
    function CauchyFast(μ::T, σ::T) where {T}
        lnorm = -log(π*σ)
        return new{T}(μ, σ, lnorm)
    end
end

@inline @fastmath function Distributions.logpdf(d::CauchyFast, x::Real)
    return d.lnorm - log1p(((x-d.μ)/d.σ)^2)
end

Distributions.quantile(d::CauchyFast, x::Real; kwargs...) = quantile(Cauchy(d.μ, d.σ), x; kwargs...)

function Distributions.cdf(d::CauchyFast, x::Real)
    return Distributions.cdf(Cauchy(d.μ, d.σ),x)
end

Distributions.zval(d::CauchyFast, x) = (x - d.μ)/d.σ


Distributions.@distr_support CauchyFast -Inf Inf

using Distributions: zval

function Distributions.logcdf(d::CauchyFast, x::Real)
    Distributions.logcdf(Cauchy(d.μ, d.σ), x)
end

function Distributions.logccdf(d::CauchyFast, x::Real)
    Distributions.logccdf(Cauchy(d.μ, d.σ), x)
end