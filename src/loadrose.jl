function hdf5_chain(dir, outname)
    dfchain, dfsum = load_chains(dir)
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


"""
    load_chains(dir)
Loads the set of csv chains and summary file assuming the structure present in parallel_main
"""
function load_chains(dir)
    files = filter(endswith(".csv"), readdir(joinpath(dir,"Chain"), join=true))
    sfile = joinpath(dir,"merged_stats.csv")
    dfsum = merge_summaries(joinpath(dir, "Stats"))
    CSV.write(sfile, dfsum)

    ind = parse.(Int,first.(splitext.(last.(split.(files, '-')))))
    sind = sortperm(ind)
    dfs = CSV.File.(files) .|> DataFrame
    for df in dfs
        df.diamdb = @. df.diam - 1/(4*log(2))*df.fwhm^2/df.diam
    end

    return dfs[sind], dfsum
end


function merge_summaries(sumdir)
    files = filter(endswith(".csv"), readdir(sumdir, join=true))
    df = CSV.File(files[1]) |> DataFrame
    for f in files[2:end]
        append!(df, CSV.File(f)|> DataFrame)
    end
    return sort!(df, :time)
end
