module EHTModelStacker

export ChainH5, SnapshotWeights, getsnapshot, lpdf, keys, getparam,
       restricttime, getsnapshot, getsnapshotdf
using Distributions
using HDF5
using StatsFuns
using KernelDensity
using CSV
using TupleVectors
using LoopVectorization
using DataFrames
using ArraysOfArrays
using SpecialFunctions: besselix, beta, erfcx, erfc
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
    schain = flatview(chain.chain)[:, 1:d.batchsize, :]
    buffer = zeros(size(schain, 2))
    #println(chain.times)
    @inbounds for i in eachindex(chain.times)
        csub = @view schain[:,:, i]
        tmp = 0.0
        ntot = 0
        for j in axes(csub,2)
            l = @view(csub[:,j])
            #println(l)
            buffer[j] = Distributions.logpdf(d.transition,l) - Distributions.logpdf(d.prior, l)
            #@assert isfinite(buffer[j]) "WTF $l"
            #ntot += 1
            #a = exp(Distributions.logpdf(d.transition,l) - Distributions.logpdf(d.prior, l))
            #println(a)
            #@assert isfinite(a) "WTF $l"
            #tmp += a
        end
        #println(mean(exp.(buffer)), " ", std(exp.(buffer)))
        ls += logsumexp(buffer)
        #ls += log(tmp)
        #println(ls)
        #println(ntot)
    end
    return ls - length(chain.times)*log(d.batchsize)
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

function ChainH5(filename::String, nsamples=2000)
    chain = h5open(filename, "r") do fid
        times = read(fid["time"])
        params = read(fid["params"])
        logz = read(fid["logz"])
        k = sortperm(parse.(Int, last.(split.(keys(fid["params"]), "scan"))))
        names = keys(fid["params"])[k]
        quant = Symbol.(keys(fid["params"]["scan1"]))
        chain = nestedview(zeros(length(quant), nsamples, length(names)),2)
        for (i,n) in enumerate(names)
            chain[i] = Array(hcat([params[n][String(k)][1:nsamples] for k in quant]...)')
        end
        ChainH5{typeof(chain), typeof(quant), typeof(times), typeof(logz)}(chain, quant, times, logz)
    end

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

function getsnapshot(c::ChainH5, i::Int)
    ChainH5{typeof(c.chain[i]), typeof(c.names), typeof(c.times[i]), typeof(c.logz[i])}(
                c.chain[i],
                c.names,
                c.times[i],
                c.logz[i],
                )
end

function getsnapshotdf(c::ChainH5, i::Int)
    df = DataFrame((c.names[n] => c.chain[i][n, :] for n in eachindex(c.names))...)
    return c.times[i], df
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
