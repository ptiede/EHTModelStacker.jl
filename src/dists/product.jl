"""
    MyProduct(dists)
Creates my version of the product distribution that I typically find had better
type stability than the usual Distributions version.

# Notes
Only logpdf is implemented so no calling rand.
"""
struct MyProduct{T} <: Distributions.ContinuousMultivariateDistribution
    dists::T
end

marginal(d::MyProduct, i) = @inbounds d.dists[i]

Base.@propagate_inbounds @inline function Distributions.logpdf(d::MyProduct, x::AbstractVector)
    acc = 0.0
    for i in eachindex(d.dists,x)
        acc += Distributions.logpdf(marginal(d, i), x[i])
    end
    return acc
end
