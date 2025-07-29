function make_hdf5_chain_rose(dir, outname)
    dfchain, dfsum = load_chains(dir)
    write2h5(dfchain, dfsum, outname)
end


"""
    load_chains(dir)
Loads the set of csv chains and summary file assuming the structure present in parallel_main
"""
function load_chains(dirs)
    chains = load_chains_1day.(dirs)
    dfs = vcat(first.(chains)...)
    dfsum = vcat(last.(chains)...)
    return dfs, dfsum
end


function load_chains_1day(dir)
    files = filter(x->endswith(x,".csv")&&startswith(basename(x), "equal_chain"), readdir(joinpath(dir,"Chain"), join=true))
    sfile = joinpath(dir,"merged_stats.csv")
    dfsum = merge_summaries(joinpath(dir, "Stats"))
    CSV.write(sfile, dfsum)

    ind = parse.(Int,first.(splitext.(last.(split.(files, '-')))))
    sind = sortperm(ind)
    dfs = CSV.File.(files) .|> DataFrame
    #for df in dfs
    #    df.img_diamdb = @. df.img_diam - 1/(4*log(2))*df.img_fwhm^2/df.img_diam
    #end

    return dfs[sind], dfsum
end


function removegains!(df)
    keep = []
    for n in names(df)
      if !startswith(String(n),"g_σ")
        push!(keep, n)
      end
    end
    select!(df, keep...)
end


function merge_summaries(sumdir)
    files = filter(endswith(".csv"), readdir(sumdir, join=true))
    df = CSV.File(files[1]) |> DataFrame
    removegains!(df)
    for f in files[2:end]
      append!(df, removegains!(CSV.File(f)|> DataFrame))
    end
    return sort!(df, :time)
end
