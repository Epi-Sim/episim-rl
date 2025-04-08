module EpiSim

# using MMCACovid19Vac
using ArgParse
using Dates, Logging, Printf
using HDF5, DataFrames, NetCDF

import JSON
import CSV


include("commands.jl")

function print_banner()
    println()
    println("========================================================================\n")
    print(raw"""
     /████████           /██  /██████  /██                       /██
    | ██_____/          |__/ /██__  ██|__/                      | ██
    | ██        /██████  /██| ██  \__/ /██ /██████/████      /██| ██
    | █████    /██__  ██| ██|  ██████ | ██| ██_  ██_  ██    |__/| ██
    | ██__/   | ██  \ ██| ██ \____  ██| ██| ██ \ ██ \ ██     /██| ██
    | ██      | ██  | ██| ██ /██  \ ██| ██| ██ | ██ | ██    | ██| ██
    | ████████| ███████/| ██|  ██████/| ██| ██ | ██ | ██ /██| ██| ██
    |________/| ██____/ |__/ \______/ |__/|__/ |__/ |__/|__/| ██|__/
              | ██                                     /██  | ██    
              | ██                                    |  ██████/    
              |__/                                     \______/     

    """)
    println()
    println("  A Julia package for simulating epidemic spreading in meta-populations\n")
    println("========================================================================\n")
end


function julia_main()::Cint
    """
    This is the entrypoint for the compiled version of EpiSim.
    """
    try
        print_banner()
        args = parse_command_line()
        command = args["%COMMAND%"]
    
        # Check if the provided command is in the list of accepted commands
        if !(command in COMMANDS)
            println("Unknown command: $command")
            println("Accepted commands are: $(join(COMMANDS, ", "))")
            return 1
        end

        log_level = args[command]["log-level"]
        set_log_level(log_level)
        println(" Starting EpiSim command $(command) with log level $log_level")

        if command == "run"
            execute_run(args[command])
        elseif command == "setup"
            execute_setup(args[command])
        elseif command == "init"
            execute_init(args[command])
        end
        @info "- Finished command: $command"

    catch e
        @error "error in main" exception=(e, catch_backtrace())
        return 1
    end
    println(" EpiSim finished successfully!")
    return 0
end

function main()
    try
        return julia_main()
    catch e
        @error "Error after julia_main" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

end # module EpiSim