abstract type AbstractEngine end
abstract type AbstractOutputFormat end

# Add to this as we add more engines
struct MMCACovid19VacEngine <: AbstractEngine end
struct MMCACovid19Engine <: AbstractEngine end

struct NetCDFFormat <: AbstractOutputFormat end
struct HDF5Format <: AbstractOutputFormat end


const ENGINES  = ["MMCACovid19Vac", "MMCACovid19"]
const COMMANDS = ["run", "setup", "init"]

const BASE_CONFIG_NAME = "config.json"
const BASE_METAPOP_NAME = "metapopulation_data.csv"


# Define a dictionary to map engine names to their types
const ENGINE_TYPES = Dict(
    "MMCACovid19Vac" => MMCACovid19VacEngine,
    "MMCACovid19" => MMCACovid19Engine
)