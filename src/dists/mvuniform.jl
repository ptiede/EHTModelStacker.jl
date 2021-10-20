"""
    MvUniform(mins, maxs)
Creates a multivariate uniform distribution in the box defined by [mins, maxs].
This is faster than calling Product, since I compute the lnorm once at creation
"""
struct MvUniform{T<:AbstractVector, N} <: Distributions.ContinuousMultivariateDistribution
    mins::T
    maxs::T
    lnorm::N
    function MvUniform(mins::T, maxs::T) where {T}
        lnorm = log(prod(maxs .- mins))
        return new{T, typeof(lnorm)}(mins, maxs, lnorm)
    end
end

function Base.rand(d::MvUniform)
    return d.mins .+ rand(length(d.mins)).*(d.maxs .- d.mins)
end

Distributions.support(d::MvUniform) = d.mins, d.maxs

@inline inbounds(d::MvUniform, x) = d.mins < x < d.maxs


@inline function Distributions.logpdf(d::MvUniform, x::AbstractVector)
    inbounds(d, x) ? -d.lnorm : -Inf
end
