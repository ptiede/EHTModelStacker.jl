module EHTModelStacker

export ChainH5, SnapshotWeights, getsnapshot, lpdf, keys, getparam, restricttime
using Distributions
using HDF5
using StatsFuns
using KernelDensity
using CSV
using TupleVectors
using LoopVectorization
using DataFrames
using ArraysOfArrays
using SpecialFunctions: besselix

include("loadrose.jl")
export make_hdf5_chain_rose
include("loadfreek.jl")
export make_hdf5_chain_freek



struct SnapshotWeights{T,P}
    transition::T
    prior::P
    batchsize::Int
end

struct ConstantWeights{T,P} end

struct ChainH5{N,C,T,Z}
    chain::C
    times::T
    logz::Z
end

struct MvNormal2D{T,C}
    μ::T
    Σ::C
end

struct MvNormalFast{T<:AbstractVector, N} <: Distributions.AbstractMvNormal
    μ::T
    Σ::T
    lnorm::N
    function MvNormalFast(μ::T, Σ::T) where {T}
        lnorm = -0.5*log(prod(Σ)) - length(μ)/2*log(2π)
        return new{typeof(μ), typeof(lnorm)}(μ, Σ, lnorm)
    end
end

struct NormalFast{T} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    σ::T
    lnorm::T
    function NormalFast(μ::T, σ::T) where {T}
        lnorm = -log(σ) - 0.5*log(2π)
        return new{T}(μ, σ, lnorm)
    end
end

@inline function Distributions.logpdf(d::NormalFast, x::Real)
    return d.lnorm - 0.5*((x-d.μ)/d.σ)^2
end


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

@inline function Distributions.logpdf(d::MvNormalFast, x::AbstractVector)
    μ,Σ = d.μ, d.Σ
    acc = zero(eltype(x))
    @turbo for i in eachindex(x)
        acc += -(x[i]-μ[i])^2/Σ[i]/2
    end
    return acc + d.lnorm
end

@inline Distributions.pdf(d::MvNormalFast, x) = exp(logpdf(d, x))


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

#=
function lpdf(d::SnapshotWeights, chain::ChainH5{K,A,B,C}) where {K<:NTuple{1}, A, B, C}
    ls = 0.0
    @inbounds for i in eachindex(chain.times)
        csub = chain.chain[i]
        tmp = 0.0
        for i in 1:chain.nsamples
            ind = i#rand(1:length(csub))
            tmp += pdf(d.transition, csub[ind])/pdf(d.prior, csub[ind])
        end
        ls += log(tmp/chain.nsamples+eps(typeof(tmp)))
    end
    return ls
end
=#


function lpdf(d::SnapshotWeights, chain::ChainH5)
    ls = 0.0
    #ind = rand(axes(flatview(chain.chain),2), d.batchsize)
    schain = @view flatview(chain.chain)[:, 1:d.batchsize, :]
    @inbounds @simd for i in eachindex(chain.times)
        csub = @view schain[:,:, i]
        tmp = 0.0
        @inbounds for l in eachcol(csub)
            #l = @view csub[:,i]
            tmp += exp(Distributions.logpdf(d.transition,l) - Distributions.logpdf(d.prior, l))
        end
        ls += log(tmp/d.batchsize+eps(typeof(tmp)))
    end
    return ls
end




@generated function getsnapshot(c::ChainH5{N}, i::Int) where {N}
    quote
        ($N=c.chain[i], time = c.times[i], logz=c.logz[i])
    end
end

function ChainH5(filename::String, quant::Symbol, nsamples=2000)
    fid = h5open(filename, "r")
    times = read(fid["time"])
    params = read(fid["params"])
    k = sortperm(parse.(Int, last.(split.(keys(fid["params"]), "scan"))))
    names = keys(params)[k]
    chain = [params[n][String(quant)][1:nsamples] for n in names]
    logz = read(fid["logz"])

    return ChainH5{quant, typeof(chain), typeof(times), typeof(logz)}(chain, times, logz)
end

function ChainH5(filename::String, quant::NTuple{N,Symbol}, nsamples=2000) where {N}
    chain = h5open(filename, "r") do fid
        times = read(fid["time"])
        params = read(fid["params"])
        logz = read(fid["logz"])
        k = sortperm(parse.(Int, last.(split.(keys(fid["params"]), "scan"))))
        names = keys(fid["params"])[k]
        chain = nestedview(zeros(length(quant), nsamples, length(names)),2)
        for (i,n) in enumerate(names)
            chain[i] = Array(hcat([params[n][String(k)][1:nsamples] for k in quant]...)')
        end
        ChainH5{quant, typeof(chain), typeof(times), typeof(logz)}(chain, times, logz)
    end
    return chain
end

function getparam(c::ChainH5{K, A, B, C}, p::Symbol) where {K, A, B, C}
    i = findfirst(x->x==p, K)
    chain = [c.chain[n][i, :] for n in 1:length(c.chain)]
    return ChainH5{K[i], typeof(chain), typeof(c.times), typeof(c.logz)}(chain, c.times, c.logz)
end

function getparam(c::ChainH5{K, A, B, C}, ps::NTuple{N,Symbol}) where {K, A, B, C, N}
    i = Int[findfirst(x->x==p, K) for p in ps]
    chain = [c.chain[n][i, :] for n in 1:length(c.chain)]
    return ChainH5{ps, typeof(chain), typeof(c.times), typeof(c.logz)}(chain, c.times, c.logz)
end



function restricttime(c::ChainH5, tmin, tmax)
    if tmin < c.times[1]
        imin = 1
    else
        imin = findfirst(x->x>tmin, c.times)
    end

    if tmax > c.times[end]
        imax = length(c.times)
    else
        imax = findfirst(x->x>tmax, c.times)-1
    end

    return ChainH5{keys(c), typeof(c.chain), typeof(c.times), typeof(c.logz)}(
                c.chain[imin:imax],
                c.times[imin:imax],
                c.logz[imin:imax],
                )
end

Base.keys(c::ChainH5{K, A, B, C}) where {K,A,B,C} = K

function flatchain(chain::ChainH5)
    return chain.chain
end




end #end
