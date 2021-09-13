module EHTModelStacker

export ChainH5, SnapshotWeights, getsnapshot, lpdf
using Distributions
using HDF5
using StatsFuns
using KernelDensity
using CSV

include("loadrose.jl")
export make_hdf5_chain_rose
include("loadfreek.jl")
export make_hdf5_chain_freek 



struct SnapshotWeights{T,P}
    transition::T
    prior::P
end

struct ConstantWeights{T,P} end

struct ChainH5{N,C,T,Z}
    chain::C
    times::T
    logz::Z
    nsamples::Int
end


function lpdf(d::SnapshotWeights, chain::ChainH5)
    ls = zero(eltype(chain.chain[1]))
    @inbounds for i in eachindex(chain.times)
        csub = @view chain.chain[i][1:chain.nsamples]
        tmp = sum(x->pdf(d.transition,x)/pdf(d.prior, x), csub)/chain.nsamples
        ls += log(tmp+eps(typeof(tmp)))
    end
    return ls
end


@generated function getsnapshot(c::ChainH5{N}, i::Int) where {N}
    quote
        ($N=c.chain[i], time = c.times[i], logz=c.logz[i])
    end
end

function ChainH5(filename::String, quant::String, nsamples::Int)
    fid = h5open(filename, "r")
    times = read(fid["time"])
    params = read(fid["params"])
    names = ["scan$i" for i in 1:length(times)]
    chain = [params[n][quant] for n in names]
    logz = read(fid["logz"])

    return ChainH5{Symbol(quant), typeof(chain), typeof(times), typeof(logz)}(chain, times, logz, nsamples)
end

end #end
