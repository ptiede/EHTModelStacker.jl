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
using SpecialFunctions: besselix, beta
using NPZ


include(joinpath(@__DIR__, "dists/dists.jl"))

include("loadrose.jl")
export make_hdf5_chain_rose
include("loadfreek.jl")
export make_hdf5_chain_freek
include("loaddpi.jl")
export make_hdf5_chain_dpi



struct SnapshotWeights{T,P}
    transition::T
    prior::P
    batchsize::Int
end

struct ConstantWeights{T,P} end

struct ChainH5{C,N,T,Z}
    chain::C
    names::N
    times::T
    logz::Z
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




function ChainH5(filename::String, quant::Symbol, nsamples=2000)
    fid = h5open(filename, "r")
    times = read(fid["time"])
    params = read(fid["params"])
    k = sortperm(parse.(Int, last.(split.(keys(fid["params"]), "scan"))))
    names = keys(params)[k]
    chain = [params[n][String(quant)][1:nsamples] for n in names]
    logz = read(fid["logz"])

    return ChainH5{typeof(chain), typeof(quant), typeof(times), typeof(logz)}(chain, quant, times, logz)
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
        ChainH5{typeof(chain), typeof(quant), typeof(times), typeof(logz)}(chain, quant, times, logz)
    end
    return chain
end

function getparam(c::ChainH5, p::Symbol)
    i = findfirst(x->x==p, c.names)
    chain = [c.chain[n][i, :] for n in 1:length(c.chain)]
    return ChainH5{typeof(chain), typeof(p), typeof(c.times), typeof(c.logz)}(chain, p, c.times, c.logz)
end

function getparam(c::ChainH5, ps::NTuple{N,Symbol}) where {N}
    i = Int[findfirst(x->x==p, c.names) for p in ps]
    chain = [c.chain[n][i, :] for n in 1:length(c.chain)]
    return ChainH5{typeof(chain), typeof(ps), typeof(c.times), typeof(c.logz)}(chain, ps, c.times, c.logz)
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

    return ChainH5{typeof(c.chain), typeof(c.names), typeof(c.times), typeof(c.logz)}(
                c.chain[imin:imax],
                c.names,
                c.times[imin:imax],
                c.logz[imin:imax],
                )
end

Base.keys(c::ChainH5) = c.names

function flatchain(chain::ChainH5)
    return chain.chain
end




end #end
