
"""
    NormalFast(μ,σ)
This creates a normal distribution that is an order of magnitude faster than the
Distributions.jl implementation since is computes the normalization value at
creation since this will be constant. This is a limited distribution though, and
only have logpdf, logcdf, and support implemented.
"""
struct NormalFast{T} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    lnorm::T
    function NormalFast(μ::T, σ::T) where {T}
        lnorm = -log(σ) - 0.5*log(2π)
        return new{T}(μ, σ, lnorm)
    end
end

function Distributions.cdf(d::NormalFast, x::Real)
    return Distributions.cdf(Normal(d.μ, d.σ),x)
end


@inline function Distributions.logpdf(d::NormalFast, x::Real)
    return d.lnorm - 0.5*((x-d.μ)/d.σ)^2
end

Distributions.@distr_support NormalFast -Inf Inf

# logcdf
function _normlogcdf(z::Real)
    if z < -one(z)
        return log(erfcx(-z * invsqrt2)/2) - abs2(z)/2
    else
        return log1p(-erfc(z * invsqrt2)/2)
    end
end

function Distributions.logcdf(d::NormalFast, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(NormalFast(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normlogcdf(z)
end

# logccdf
function _normlogccdf(z::Real)
    if z > one(z)
        return log(erfcx(z * invsqrt2)/2) - abs2(z)/2
    else
        return log1p(-erfc(-z * invsqrt2)/2)
    end
end

function Distributions.logccdf(d::NormalFast, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(NormalFast(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    return _normlogccdf(z)
end

# cdf
_normcdf(z::Real) = erfc(-z * invsqrt2)/2

"""
    WrappedNormal(μ, σ)
Creates a normal distribution that is 2π wrapped. Note this gets the normalization wrong!
"""
struct WrappedNormal{T, S} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    lnorm::S
    function WrappedNormal(μ::T, σ::T) where {T}
        lnorm = -log(σ) - 0.5*log(2π)
        return new{T, typeof(lnorm)}(μ, σ, lnorm)
    end
end

function Distributions.logpdf(d::WrappedNormal, x::Real)
    s,c = sincos(x-d.μ)
    dθ = atan(s,c)
    return -0.5*abs2(dθ/d.σ) + d.lnorm
end
