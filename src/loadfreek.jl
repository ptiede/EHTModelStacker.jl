function hdf5_chain_freek(dir, scanfile, outname)
    dfchain, dfsum = load_chains_freek(dir, scanfile)
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


function load_chains_freek(dir, scanfile)
    files = filter(endswith(".npy"), readdir(dir,join=true))
    sfiles = replace.(files, Ref("_samples.npy"=>"_summary.txt"))
    pfiles = replace.(files, Ref("_samples.npy"=>"_params.txt"))
    ind = parse.(Int, first.(split.(last.(split.(files, Ref("scan="))), Ref("_"))))
    sind = sortperm(ind)
    dfsum = readscanfilefreek(scanfile)
    logz = Float64[readlogzfreek(s) for s in sfiles]
    dfsum.logz = logz

    pnames = readparamnamesfreek(pfiles[1])
    dfs = DataFrame[]
    for i in eachindex(files)
        push!(dfs, convert_freekchain(files[i], pnames))
    end
    return dfs[sind], dfsum
end

function readscanfilefreek(scanfile)
    df = CSV.File(scanfile,
             delim=' ',
             comment="#",
             header=[:scan, :time, :foo1, :foo2],
             select=[2]) |> DataFrame
    return df
end

function readlogzfreek(sfile)
    logz = open(sfile, "r") do io
        lines = readlines(io)
        logz = parse(Float64, lines[end])
        return logz
    end
    return logz
end

function readparamnamesfreek(pfile)
    open(pfile, "r") do io
        lines = readlines(io)[2:end]
        slines = split.(lines)
        names = String[]
        for l in slines
            if length(l) == 4
                push!(names, l[1])
            end
        end
        return names
    end
end


"""
    convert_freekchain(file, pnames)

Converts the parameter names in freek's files to the ones used by my ROSE based method.
"""
function convert_freekchain(file, pnames)
    params = npzread(file)
    pfix = copy(pnames)
    for (i,p) in enumerate(pnames)
        if startswith(p, "beta")&&endswith(p,"abs")
            order = p[5]
            pfix[i] = "ma"*order
        elseif startswith(p, "beta")&&endswith(p,"arg")
            order = p[5]
            pfix[i] = "mp"*order
        elseif p=="ff"
            pfix[i]="floor"
        elseif p=="alpha"
            pfix[i]="fwhm"
        elseif p=="F0"
            pfix[i]="f"
        elseif p=="stretch-PA"
            pfix[i]="ξτ"
        end
    end
    df = DataFrame([params[:,i] for i in 1:length(pfix)], Symbol.(pfix))
    if "stretch" ∈ names(df)
        df.diam = df[!,:d].*sqrt.(df[!,:stretch])
        df.diamdb = @. df.diam - 1/(4*log(2))*df.fwhm^2/df.diam
        df.τ = @. 1-1/df[!,:stretch]
    else
        df.diam = df[!,:d]
        df.diamdb = @. df.diam - 1/(4*log(2))*df.fwhm^2/df.diam
    end
    return df
end
