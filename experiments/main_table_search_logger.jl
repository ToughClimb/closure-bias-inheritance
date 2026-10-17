"""Read-only search-cycle audit for the Table 4 rerun."""
module MainTableSearchAudit

using Logging
import SymbolicRegression

mutable struct SearchLogger <: SymbolicRegression.AbstractSRLogger
    io::IOStream
    started::Float64
    total_cycles::Int
    remaining::Vector{Int}
    evaluations::Float64
    calls::Int
    base_logger::AbstractLogger
end

const REGISTRY = Dict{String,SearchLogger}()

function create(path::String, total_cycles::Int)
    @assert !isfile(path) "Refuse to overwrite an existing cycle audit"
    io = open(path, "w")
    println(io, "callback,unix_seconds,elapsed_seconds,completed_population_cycles,remaining_population_cycles,evaluations")
    flush(io)
    logger = SearchLogger(io, time(), total_cycles, [total_cycles], 0.0, 0, ConsoleLogger(stderr))
    REGISTRY[path] = logger
    return logger
end

SymbolicRegression.get_logger(logger::SearchLogger) = logger.base_logger

function SymbolicRegression.logging_callback!(logger::SearchLogger; state, datasets, ropt, options)
    logger.remaining = copy(state.cycles_remaining)
    logger.evaluations = sum(sum, state.num_evals)
    logger.calls += 1
    now = time()
    remaining = sum(logger.remaining)
    println(logger.io, join((logger.calls, now, now-logger.started,
                            logger.total_cycles-remaining, remaining, logger.evaluations), ','))
    flush(logger.io)
    return nothing
end

function finish(logger::SearchLogger)
    if isopen(logger.io)
        flush(logger.io)
        close(logger.io)
    end
    return nothing
end

function statistics(path::String)
    logger = REGISTRY[path]
    return (logger.total_cycles, sum(logger.remaining), logger.evaluations, logger.calls)
end

end
