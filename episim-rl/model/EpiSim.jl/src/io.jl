include("common.jl")

function update_config!(config, cmd_line_args)
    # Define dictionary containing epidemic parameters

    # overwrite config with command line
    if cmd_line_args["start-date"] !== nothing
        config["simulation"]["start_date"] = cmd_line_args["start-date"]
    end
    if cmd_line_args["end-date"] !== nothing
        config["simulation"]["end_date"] = cmd_line_args["end-date"]
    end
    if cmd_line_args["export-compartments-time-t"] !== nothing
        config["simulation"]["export_compartments_time_t"] = cmd_line_args["export-compartments-time-t"]
    end
    if cmd_line_args["export-compartments-full"] == true
        config["simulation"]["export_compartments_full"] = true
    end
    if cmd_line_args["initial-condition"] !== nothing
        config["simulation"]["initial_condition_filename"] = cmd_line_args["initial-condition"]
    end

    nothing
end


const OUTPUT_FORMATS = Dict("netcdf" => NetCDFFormat(), "hdf5" => HDF5Format())

get_output_format(output_format::String) = get(OUTPUT_FORMATS, output_format, NetCDFFormat())
get_output_format_str(output_format::AbstractOutputFormat) = findfirst(==(output_format), OUTPUT_FORMATS)

function save_full(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, output_format::Union{String,AbstractOutputFormat}; kwargs...)
    
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_full(engine, epi_params, population, output_path, format; kwargs...)
end

function _save_full(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, ::NetCDFFormat; 
    G_coords=String[], M_coords=String[], T_coords=String[])
    
    
    filename = joinpath(output_path, "compartments_full.nc")
    @info "- Saving full simulation output in NetCDF: $filename"
    try
        MMCACovid19Vac.save_simulation_netCDF(epi_params, population, filename; G_coords, M_coords, T_coords)
    catch e
        @error "Error saving simulation output" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"
end

function _save_full(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, ::HDF5Format; kwargs...)

    filename = joinpath(output_path, "compartments_full.h5")
    @info "- Saving full simulation output in HDF5: $filename"
    MMCACovid19Vac.save_simulation_hdf5(epi_params, population, filename)
end

function save_time_step(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, output_format::Union{String,AbstractOutputFormat}, 
    export_time_t::Int, export_date::Date; kwargs...)
    
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_time_step(engine, epi_params, population, output_path, format, export_time_t, export_date; kwargs...)
end

function _save_time_step(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, ::HDF5Format, export_compartments_time_t::Int, 
    export_date::Date; kwargs...) 
    
    filename = joinpath(output_path, "compartments_t_$(export_date).h5")

    @info "- Saving simulation state at time step: $(export_compartments_time_t) ($(export_date))"
    @info "\t- filename: $(filename)"
    MMCACovid19Vac.save_simulation_hdf5(epi_params, population, filename; 
                        export_time_t = export_compartments_time_t)
end

function _save_time_step(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String, ::NetCDFFormat, export_compartments_time_t::Int, 
    export_date::Date; G_coords=String[], M_coords=String[]) 

    filename = joinpath(output_path, "compartments_t_$(export_date).nc")
    @info "- Saving simulation state at time step: $(export_compartments_time_t) ($(export_date))"
    @info "\t* Filename: $(filename)"
    try
        G = population.G
        M = population.M
        V = epi_params.V
        V_coords = epi_params.VaccLabels

        if isnothing(G_coords) != G
            G_coords = collect(1:G)
        end
        if isnothing(M_coords)
            M_coords = collect(1:M)
        end

        g_dim = NcDim("G", G, atts=Dict("description" => "Age strata", "Unit" => "unitless"), values=G_coords, unlimited=false)
        m_dim = NcDim("M", M, atts=Dict("description" => "Region", "Unit" => "unitless"), values=M_coords, unlimited=false)
        v_dim = NcDim("V", V, atts=Dict("description" => "Vaccination status", "Unit" => "unitless"), values=V_coords, unlimited=false)
        dimlist = [g_dim, m_dim, v_dim]

        S  = NcVar("S" , dimlist; atts=Dict("description" => "Suceptibles"), t=Float64, compress=-1)
        E  = NcVar("E" , dimlist; atts=Dict("description" => "Exposed"), t=Float64, compress=-1)
        A  = NcVar("A" , dimlist; atts=Dict("description" => "Asymptomatic"), t=Float64, compress=-1)
        I  = NcVar("I" , dimlist; atts=Dict("description" => "Infected"), t=Float64, compress=-1)
        PH = NcVar("PH", dimlist; atts=Dict("description" => "Pre-hospitalized"), t=Float64, compress=-1)
        PD = NcVar("PD", dimlist; atts=Dict("description" => "Pre-deceased"), t=Float64, compress=-1)
        HR = NcVar("HR", dimlist; atts=Dict("description" => "Hospitalized-good"), t=Float64, compress=-1)
        HD = NcVar("HD", dimlist; atts=Dict("description" => "Hospitalized-bad"), t=Float64, compress=-1)
        R  = NcVar("R" , dimlist; atts=Dict("description" => "Recovered"), t=Float64, compress=-1)
        D  = NcVar("D" , dimlist; atts=Dict("description" => "Dead"), t=Float64, compress=-1)
        CH  = NcVar("CH" , dimlist; atts=Dict("description" => "Confined"), t=Float64, compress=-1)
        varlist = [S, E, A, I, PH, PD, HR, HD, R, D, CH]
        
        data_dict = Dict()
        data_dict["S"]  = epi_params.ρˢᵍᵥ[ :, :, export_compartments_time_t, :]  .* population.nᵢᵍ
        data_dict["E"]  = epi_params.ρᴱᵍᵥ[ :, :, export_compartments_time_t, :]  .* population.nᵢᵍ
        data_dict["A"]  = epi_params.ρᴬᵍᵥ[ :, :, export_compartments_time_t, :]  .* population.nᵢᵍ
        data_dict["I"]  = epi_params.ρᴵᵍᵥ[ :, :, export_compartments_time_t, :]  .* population.nᵢᵍ
        data_dict["PH"] = epi_params.ρᴾᴴᵍᵥ[ :, :, export_compartments_time_t, :] .* population.nᵢᵍ
        data_dict["PD"] = epi_params.ρᴾᴰᵍᵥ[ :, :, export_compartments_time_t, :] .* population.nᵢᵍ
        data_dict["HR"] = epi_params.ρᴴᴿᵍᵥ[ :, :, export_compartments_time_t, :] .* population.nᵢᵍ
        data_dict["HD"] = epi_params.ρᴴᴰᵍᵥ[ :, :, export_compartments_time_t, :] .* population.nᵢᵍ
        data_dict["R"]  = epi_params.ρᴿᵍᵥ[ :, :, export_compartments_time_t, :]  .* population.nᵢᵍ
        data_dict["D"]  = epi_params.ρᴰᵍᵥ[ :, :, export_compartments_time_t, :]  .* population.nᵢᵍ
        data_dict["CH"] = epi_params.CHᵢᵍᵥ[ :, :, export_compartments_time_t, :] .* population.nᵢᵍ
        
        isfile(filename) && rm(filename)

        NetCDF.create(filename, varlist, mode=NC_NETCDF4)
        for (var_label, data) in data_dict
            ncwrite(data, filename, var_label)
        end
        
    catch e
    
        @error "Error saving time step" exception=(e, catch_backtrace())
    end
    @info "- Done saving"
end


function save_observables(engine::MMCACovid19VacEngine, 
    epi_params::MMCACovid19Vac.Epidemic_Params,
    population::MMCACovid19Vac.Population_Params,
    output_path::String; 
    G_coords=String[], M_coords=String[], T_coords=String[])

    filename = joinpath(output_path, "observables.nc")
    @info "- Storing simulation observables output in NetCDF: $filename"
    try
        MMCACovid19Vac.save_observables_netCDF(epi_params, population, filename; G_coords, M_coords, T_coords)
    catch e
        @error "Error saving simulation observables" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"
end




function save_full(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, output_format::Union{String,AbstractOutputFormat}; kwargs...)
    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_full(engine, epi_params, population, output_path, format; kwargs...)
end

function _save_full(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, ::NetCDFFormat; G_coords=String[], M_coords=String[], T_coords=String[])
    
    
    filename = joinpath(output_path, "compartments_full.nc")
    @info "- Saving full simulation output in NetCDF: $filename"
    try
        G = population.G
        M = population.M
        T = epi_params.T

        if length(G_coords) != G
            G_coords = collect(1:G)
        end
        if length(M_coords) != M
            M_coords = collect(1:M)
        end
        if length(T_coords) != T
            T_coords = collect(1:T) 
        end
        
        g_dim = NcDim("G", G, atts=Dict("description" => "Age strata", "Unit" => "unitless"), values=G_coords, unlimited=false)
        m_dim = NcDim("M", M, atts=Dict("description" => "Region", "Unit" => "unitless"), values=M_coords, unlimited=false)
        t_dim = NcDim("T", T, atts=Dict("description" => "Time", "Unit" => "unitless"), values=T_coords, unlimited=false)
        dimlist = [g_dim, m_dim, t_dim]

        S  = NcVar("S" , dimlist; atts=Dict("description" => "Suceptibles"), t=Float64, compress=-1)
        E  = NcVar("E" , dimlist; atts=Dict("description" => "Exposed"), t=Float64, compress=-1)
        A  = NcVar("A" , dimlist; atts=Dict("description" => "Asymptomatic"), t=Float64, compress=-1)
        I  = NcVar("I" , dimlist; atts=Dict("description" => "Infected"), t=Float64, compress=-1)
        PH = NcVar("PH", dimlist; atts=Dict("description" => "Pre-hospitalized"), t=Float64, compress=-1)
        PD = NcVar("PD", dimlist; atts=Dict("description" => "Pre-deceased"), t=Float64, compress=-1)
        HR = NcVar("HR", dimlist; atts=Dict("description" => "Hospitalized-good"), t=Float64, compress=-1)
        HD = NcVar("HD", dimlist; atts=Dict("description" => "Hospitalized-bad"), t=Float64, compress=-1)
        R  = NcVar("R" , dimlist; atts=Dict("description" => "Recovered"), t=Float64, compress=-1)
        D  = NcVar("D" , dimlist; atts=Dict("description" => "Dead"), t=Float64, compress=-1)
        CH = NcVar("CH", dimlist; atts=Dict("description" => "Confined"), t=Float64, compress=-1)

        varlist = [S, E, A, I, PH, PD, HR, HD, R, D, CH]

        data_dict = Dict()
        data_dict["S"]  = epi_params.ρˢᵍ  .* population.nᵢᵍ
        data_dict["E"]  = epi_params.ρᴱᵍ  .* population.nᵢᵍ
        data_dict["A"]  = epi_params.ρᴬᵍ  .* population.nᵢᵍ
        data_dict["I"]  = epi_params.ρᴵᵍ  .* population.nᵢᵍ
        data_dict["PH"] = epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
        data_dict["PD"] = epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
        data_dict["HR"] = epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
        data_dict["HD"] = epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
        data_dict["R"]  = epi_params.ρᴿᵍ  .* population.nᵢᵍ
        data_dict["D"]  = epi_params.ρᴰᵍ  .* population.nᵢᵍ
        data_dict["CH"] = epi_params.CHᵢᵍ .* population.nᵢᵍ

        isfile(filename) && rm(filename)

        NetCDF.create(filename, varlist, mode=NC_NETCDF4)
        for (var_label, data) in data_dict
            ncwrite(data, filename, var_label)
        end
    catch e
        @error "Error saving simulation output" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"

end

function create_compartments_array(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params)
    G = population.G
    M = population.M
    T = epi_params.T
    N = 11

    compartments = zeros(Float64, G, M, T, N);
    compartments[:, :, :, 1]  .= epi_params.ρˢᵍ .* population.nᵢᵍ
    compartments[:, :, :, 2]  .= epi_params.ρᴱᵍ .* population.nᵢᵍ
    compartments[:, :, :, 3]  .= epi_params.ρᴬᵍ .* population.nᵢᵍ
    compartments[:, :, :, 4]  .= epi_params.ρᴵᵍ .* population.nᵢᵍ
    compartments[:, :, :, 5]  .= epi_params.ρᴾᴴᵍ .* population.nᵢᵍ
    compartments[:, :, :, 6]  .= epi_params.ρᴾᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 7]  .= epi_params.ρᴴᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 8]  .= epi_params.ρᴴᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 9]  .= epi_params.ρᴿᵍ .* population.nᵢᵍ
    compartments[:, :, :, 10] .= epi_params.ρᴰᵍ .* population.nᵢᵍ
    compartments[:, :, :, 11] .= epi_params.CHᵢᵍ .* population.nᵢᵍ

    return compartments
end

function _save_full(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, ::HDF5Format; kwargs...)
    
    filename = joinpath(output_path, "compartments_full.h5")
    @info "- Storing full simulation output in HDF5: $filename"
    compartments = create_compartments_array(engine, epi_params, population)

    isfile(filename) && rm(filename)
    h5open(filename, "w") do file
        write(file, "data", compartments[:,:,:,:,:])
    end
end


function save_time_step(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params,
    population::MMCAcovid19.Population_Params,
    output_path::String, output_format::Union{String,AbstractOutputFormat}, 
    export_time_t::Int, export_date::Date; kwargs...)

    format = output_format isa String ? get_output_format(output_format) : output_format
    _save_time_step(engine, epi_params, population, output_path, format, export_time_t, export_date; kwargs...)
end


function _save_time_step(engine::MMCACovid19Engine,
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, ::NetCDFFormat, export_time_t::Int, export_date::Date; 
    G_coords=String[], M_coords=String[])

    filename = joinpath(output_path, "compartments_t_$(export_date).nc")
    @info "- Saving simulation state at time step: $(export_time_t) ($(export_date))"
    @info "\t* Filename: $(filename)"
    
    try
        G = population.G
        M = population.M

        if isnothing(G_coords)
            G_coords = collect(1:G)
        end
        if isnothing(M_coords)
            M_coords = collect(1:M)
        end

        g_dim = NcDim("G", G, atts=Dict("description" => "Age strata", "Unit" => "unitless"), values=G_coords, unlimited=false)
        m_dim = NcDim("M", M, atts=Dict("description" => "Region", "Unit" => "unitless"), values=M_coords, unlimited=false)
        dimlist = [g_dim, m_dim]
        
        S  = NcVar("S" , dimlist; atts=Dict("description" => "Suceptibles"), t=Float64, compress=-1)
        E  = NcVar("E" , dimlist; atts=Dict("description" => "Exposed"), t=Float64, compress=-1)
        A  = NcVar("A" , dimlist; atts=Dict("description" => "Asymptomatic"), t=Float64, compress=-1)
        I  = NcVar("I" , dimlist; atts=Dict("description" => "Infected"), t=Float64, compress=-1)
        PH = NcVar("PH", dimlist; atts=Dict("description" => "Pre-hospitalized"), t=Float64, compress=-1)
        PD = NcVar("PD", dimlist; atts=Dict("description" => "Pre-deceased"), t=Float64, compress=-1)
        HR = NcVar("HR", dimlist; atts=Dict("description" => "Hospitalized-good"), t=Float64, compress=-1)
        HD = NcVar("HD", dimlist; atts=Dict("description" => "Hospitalized-bad"), t=Float64, compress=-1)
        R  = NcVar("R" , dimlist; atts=Dict("description" => "Recovered"), t=Float64, compress=-1)
        D  = NcVar("D" , dimlist; atts=Dict("description" => "Dead"), t=Float64, compress=-1)
        CH = NcVar("CH", dimlist; atts=Dict("description" => "Confined"), t=Float64, compress=-1)
        
        varlist = [S, E, A, I, PH, PD, HR, HD, R, D, CH]

        data_dict = Dict()
        data_dict["S"]  = epi_params.ρˢᵍ[ :, :, export_time_t]  .* population.nᵢᵍ
        data_dict["E"]  = epi_params.ρᴱᵍ[ :, :, export_time_t]  .* population.nᵢᵍ
        data_dict["A"]  = epi_params.ρᴬᵍ[ :, :, export_time_t]  .* population.nᵢᵍ
        data_dict["I"]  = epi_params.ρᴵᵍ[ :, :, export_time_t]  .* population.nᵢᵍ
        data_dict["PH"] = epi_params.ρᴾᴴᵍ[ :, :, export_time_t] .* population.nᵢᵍ
        data_dict["PD"] = epi_params.ρᴾᴰᵍ[ :, :, export_time_t] .* population.nᵢᵍ
        data_dict["HR"] = epi_params.ρᴴᴿᵍ[ :, :, export_time_t] .* population.nᵢᵍ
        data_dict["HD"] = epi_params.ρᴴᴰᵍ[ :, :, export_time_t] .* population.nᵢᵍ
        data_dict["R"]  = epi_params.ρᴿᵍ[ :, :, export_time_t]  .* population.nᵢᵍ
        data_dict["D"]  = epi_params.ρᴰᵍ[ :, :, export_time_t]  .* population.nᵢᵍ
        data_dict["CH"] = epi_params.CHᵢᵍ[ :, :, export_time_t] .* population.nᵢᵍ

        isfile(filename) && rm(filename)

        NetCDF.create(filename, varlist, mode=NC_NETCDF4)
        for (var_label, data) in data_dict
            ncwrite(data, filename, var_label)
        end

    catch e
        @error "Error saving time step" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"    
end

function _save_time_step(engine::MMCACovid19Engine,
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String, ::HDF5Format, export_time_t::Int, export_date::Date; kwargs...) 
    
    filename = joinpath(output_path, "compartments_t_$(export_date).h5")
    @info "- Saving simulation state at time step: $(export_time_t) ($(export_date))"
    @info "\t* Filename: $(filename)"
    
    compartments = create_compartments_array(engine, epi_params, population)

    isfile(filename) && rm(filename)
    h5open(filename, "w") do file
        write(file, "data", compartments[:,:,export_time_t,:])
    end
    @info "- Done saving time step"
end


function save_observables(engine::MMCACovid19Engine, 
    epi_params::MMCAcovid19.Epidemic_Params, 
    population::MMCAcovid19.Population_Params,
    output_path::String; 
    
    G_coords=String[], M_coords=String[], T_coords=String[])

    filename = joinpath(output_path, "observables.nc")
    @info "- Storing simulation observables output in NetCDF: $filename"
    try
        G = population.G
        M = population.M
        T = epi_params.T
    
        if length(G_coords) != G
            G_coords = collect(1:G)
        end
        if length(M_coords) != M
            M_coords = collect(1:M)
        end
        if length(T_coords) != T
            T_coords = collect(1:T) 
        end
    
        g_dim = NcDim("G", G, atts=Dict("description" => "Age strata", "Unit" => "unitless"), values=G_coords, unlimited=false)
        m_dim = NcDim("M", M, atts=Dict("description" => "Region", "Unit" => "unitless"), values=M_coords, unlimited=false)
        t_dim = NcDim("T", T, atts=Dict("description" => "Time", "Unit" => "unitless"), values=T_coords, unlimited=false)
        dimlist = [g_dim, m_dim, t_dim]
    
        newI  = NcVar("new_infected" , dimlist; atts=Dict("description" => "Daily infections"), t=Float64, compress=-1)
        newH  = NcVar("new_hospitalized" , dimlist; atts=Dict("description" => "Daily hospitalizations"), t=Float64, compress=-1)
        newD  = NcVar("new_deaths" , dimlist; atts=Dict("description" => "Daily deaths"), t=Float64, compress=-1)
        R_eff = NcVar("R_eff" , dimlist; atts=Dict("description" => "Effective reproduction number over an infectious period of 14 days"), t=Float64, compress=-1)
        varlist = [newI, newH, newD, R_eff]
     
        data_dict = Dict()
        data_dict["new_infected"] = (epi_params.ρᴬᵍ  .* population.nᵢᵍ) .* epi_params.αᵍ
            
        hosp_rates = epi_params.μᵍ .* (1 .- epi_params.θᵍ) .* epi_params.γᵍ
        hosp_rates = reshape(hosp_rates, G, 1, 1)
            
        data_dict["new_hospitalized"] = ((epi_params.ρᴵᵍ  .* population.nᵢᵍ) .* hosp_rates)
            
        D = epi_params.ρᴰᵍ
        data_dict["new_deaths"] = zeros(size(D))
        data_dict["new_deaths"][:, :, 2:end] = diff((D .* population.nᵢᵍ), dims=3)

        data_dict["R_eff"] = zeros(Float64, G, M, T)
        τ = 14
        Rᵢᵍ_eff = MMCAcovid19.compute_R_eff_matrix(epi_params, population, τ)[1]
        data_dict["R_eff"][:, :, τ+1:end] = Rᵢᵍ_eff
        
        isfile(filename) && rm(filename)
        NetCDF.create(filename, varlist, mode=NC_NETCDF4)
        for (var_label, data) in data_dict
            ncwrite(data, filename, var_label)
        end
    catch e
        @error "Error saving simulation observables" exception=(e, catch_backtrace())
        rethrow(e)
    end
    @info "- Done saving"
end