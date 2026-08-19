#!/usr/bin/env -S julia --project
# =============================================================================
# Split examples/pairs_full.txt into N roughly-equal chunks.
#
# Usage:
#   julia --project examples/split_pairs.jl <N>
#
# Produces examples/pairs_chunk_{00..N-1}.txt; each file is a single line
# of comma-separated i-j pairs readable by compute_ClGR --pairs-file=.
# =============================================================================

const IN  = joinpath(@__DIR__, "pairs_full.txt")
const OUTDIR = @__DIR__

function main()
    length(ARGS) == 1 || error("usage: julia split_pairs.jl <N>")
    N = parse(Int, ARGS[1])
    N ≥ 1 || error("N must be ≥ 1")
    isfile(IN) || error("missing $IN (run generate_spherex_tracers.jl first)")

    raw = strip(read(IN, String))
    pairs = split(raw, ",")
    npairs = length(pairs)

    # Distribute remainder across first (npairs % N) chunks so sizes differ
    # by at most 1.
    base = div(npairs, N)
    rem  = mod(npairs, N)

    idx = 1
    for k in 0:N-1
        sz = base + (k < rem ? 1 : 0)
        chunk = pairs[idx : idx + sz - 1]
        idx += sz
        path = joinpath(OUTDIR, string("pairs_chunk_", lpad(k, 2, '0'), ".txt"))
        open(path, "w") do io
            print(io, join(chunk, ","))
        end
        println("  chunk $(lpad(k,2,'0')): $(sz) pairs → $(basename(path))")
    end
    @assert idx - 1 == npairs
    println("\nTotal $npairs pairs split into $N chunks.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
