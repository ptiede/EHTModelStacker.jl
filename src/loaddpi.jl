function make_hdf5_chain_dpi(dir, scanfile, outname)
    dfchain, dfsum = load_chain_dpi(dir, scanfile)
    write2h5(dfchain, dfsum, outname)
end

function readelbo(efile, order)
    elbos = -npzread(efile)[order]
end



function load_chain_dpi(dir, scanfile; efile = joinpath(dirname(dirname(dirname(dir))), "evidence_lower_bound.npy"))
    files = filter(endswith(".npy"), readdir(dir, join=true))
    dfscan = readscanfile(scanfile)
    scans = parse.(Int, first.(split.(last.(split.(files, Ref("postsamples"))), "_")))
    order = parse(Int, last(split(dir, "mring"))[1])
    println("I think this dir is a m=$order m-ring")

    #this is the elbo up to log(normalization)
    elbo = readelbo(efile, order)
    # Just make logz the average of the elbo because He only saved the total elbo
    logz = fill(elbo/length(scans), length(scans))

    # Read in scan file and cut all timestamps that aren't included in fits
    dfscan = filter!(:scan=> x->x∈scans, readscanfile(scanfile))
    dfscan.logz = logz

    dfchain = readdpichain.(files, order)

    return dfchain, dfscan

end

function readdpichain(file, order)
    chain = npzread(file)[1:5000, :]
    df = DataFrame(img_diam=chain[:,1],
                   img_fwhm = chain[:,2]*sqrt(2*log(2)),
                   img_ma_1 = chain[:,3]/2,
                   img_mp_1 = deg2rad.(chain[:,4]),
                   )
    starti = 4
    for (i,o) in enumerate(2:order)
        insertcols!(df, "img_ma_$o" => chain[:,starti+i])
        insertcols!(df, "img_mp_$o" => deg2rad.(chain[:,starti+1+i]))
        starti+=2
    end
    df.img_floor = chain[:,end-1]
    df.img_dg = chain[:,end]*sqrt(2*log(2))

    return df
end
