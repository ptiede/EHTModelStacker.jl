module EHTModelStacker

export ChainH5, SnapshotWeights, getsnapshot, lpdf, keys, getparam, restricttime
using Distributions
using HDF5
using StatsFuns
using CSV
using DataFrames
using ArraysOfArrays
using SpecialFunctions: besselix, beta
using OhMyThreads
using StableRNGs


include(joinpath(@__DIR__, "dists/dists.jl"))


struct SnapshotWeights{T,P,E}
    transition::T
    prior::P
    batchsize::Int
    scheduler::E
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
    #ind = rand(axes(flatview(chain.chain),2), d.batchsize)
    schain = @view flatview(chain.chain)[:, 1:d.batchsize, :]
    ls = tmapreduce(+, eachindex(chain.times); scheduler=d.scheduler) do i
        csub = @view schain[:,:, i]
        tmp = 0.0
        @inbounds for l in eachcol(csub)
            tmp += exp(Distributions.logpdf(d.transition,l) - Distributions.logpdf(d.prior, l))
        end
        return log(tmp/d.batchsize+eps(typeof(tmp)))
    end
    return ls
end

function write2h5(dfchain, dfsum, outname)
    h5open(outname, "w") do fid
        fid["time"] = dfsum[:,:time]
        fid["logz"] = dfsum[:,:logz]
        pid = create_group(fid, "params")
        for i in 1:length(dfchain)
            sid = create_group(pid, "scan$i")
            keys = names(dfchain[i])
            for k in keys
                write(sid, k, dfchain[i][:,Symbol(k)])
            end
        end
    end
end





function _load_single(filename, quant, nsamples)
    h5open(filename, "r") do fid
        times = read(fid["time"])
        params = read(fid["params"])
        logz = read(fid["logz"])
        k = sortperm(parse.(Int, last.(split.(keys(fid["params"]), "scan"))))
        names = keys(fid["params"])[k]
        if isnothing(quant)
            quant = Symbol.(keys(params[first(names)]))
        end
        rng = StableRNG(42)
        chain = nestedview(zeros(length(quant), nsamples, length(names)),2)
        ns = rand(rng, 1:length(params[first(names)][String(quant[1])]), nsamples) # Grab random samples
        for (i,n) in enumerate(names)
            chain[i] = Array(hcat([params[n][String(k)][ns] for k in quant]...)')
        end
        return chain, times, quant, logz
    end
end

function ChainH5(files::Vector{String}; quant = nothing, nsamples=2000)
    chain,  times, logz = _load_single(files[1], quant, nsamples)
    for f in files[2:end]
        c, t, q, lz = _load_single(f, quant, nsamples)
        chain = vcat(chain, c)
        times = vcat(times, t)
        logz  = vcat(logz, lz)
    end
    ChainH5{typeof(chain), typeof(quant), typeof(times), typeof(logz)}(chain, quant, times, logz)
end


function ChainH5(filename::String; quant=nothing, nsamples=2000)
    chain, times, q, logz = _load_single(filename, quant, nsamples)
    ChainH5{typeof(chain), typeof(q), typeof(times), typeof(logz)}(chain, q, times, logz)
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
