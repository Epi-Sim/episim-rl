```
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
```
## A Julia package for simulating epidemic spreading in metapopulations

_EpiSim.jl_ is a general interface to access different simulation engines/models. In this sense, _EpiSim.jl_ works as a stand-alone simulator that allows running simulations using different engines/models. Currently, it implements MMCA approach for simulating an extended SEIR in a meta-population with different agent types which can be used to model different age strata [1-4]. The different populations are connect through a network that represent the daily mobility patterns of agents that commute to work to different regions.

Currently supported engines:
* **MMCAcovid19** Julia package, [https://docs.juliahub.com/MMCAcovid19/]([url](https://docs.juliahub.com/MMCAcovid19/)) [2].
* **MMCAcovid19-Vac** Julia package, [https://github.com/Epi-Sim/MMCACovid19Vac.jl]([url](https://github.com/Epi-Sim/MMCACovid19Vac.jl)) [3]

As a novel feature, _EpiSim.jl_ uses a standard configuration format for setting or defining specific instances of a model, for instance, defining the number and sizes of the different metapopulations, providing the structure of the mobility network and setting values for the epidemiological transition rates.

Additionally, EpiSim.jl uses the NetCDF format to store simulation outputs. [NetCDF]([url](https://www.unidata.ucar.edu/software/netcdf/)) enables storing multi-dimensional arrays with labelled coordinates. Working with multi-dimensional arrays in a standard format helps the interoperability, the post-processing of the results and the exchange of results between modellers. The following figures show an example of the multi-dimensional structure of a simulation:

<img src="https://github.com/user-attachments/assets/f4a44223-3377-47d5-9212-e79252300343" width="600">

  
EpiSim.jl also provides a standard configuration format for defining a model and a simple set of command-line scripts to generate configuration templates and run the model.


### Installing and Running

To install EpiSim.jl, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/EpiSim.jl.git
   cd EpiSim.jl
   ```

2. Compiling (optional):
   ```
   julia ./install.jl -c -i
   ```

The compilation process (`install.jl`) creates a single precompiled executable:

- Builds the application in a `build` folder
- Creates a symlink named `episim` in the target directory (default is the current directory)

After installation, you can run EpiSim using the `episim` command.

### Using EpiSim.jl

EpiSim works as a command line frontend to launch simulations. It provides a simple JSON-based config format to define an instance of a model. The config format included a common or core set of parameters and specific parameters required by the different engines supported by EpiSim.jl. An example is given at `models/mitma/config_MMCACovid19.json`


### The `epiconfig.json` format
```
{
	"simulation": {
        "engine": "MMCACovid19",
		"start_date": "2020-03-10",
		"end_date": "2020-04-15",
		"save_full_output": true,
		"save_time_step": null,
		"output_folder": "output",
		"output_format": "netcdf"
	},
	"data": {
		# Engine specific
        # path to files with input data, e.g. metapopulation, mobility matrix, etc.
	},
	"epidemic_params": {
		# Engine specific
        # Epidemic parameters e.g. infection and recovery rates.
    },
	"population_params": {
        # Engine specific
        # Population parameter, e.g. average contacts, age-specific contact matrix.
    }
}
```

```bash
episim -e MMCACovid19Vac run -c models/mitma/config.json -d models/mitma -i runs
```


# Brief history

## MMCAcovid19
This package [MMCAcovid19](https://github.com/jtmatamalas/MMCAcovid19), written in the [Julia](https://julialang.org) language, implements the epidemic model for COVID-19 developed by a group of researchers from [Universitat Rovira i Virgili](https://www.urv.cat) and [Universidad de Zaragoza](http://unizar.es) [[1](#References-1)]. The model makes use of a Microscopic Markov Chain Approach (MMCA) to describe mathematically the dynamics of a so-called metapopulation model of epidemic spreading [[2-4](#References-1)]. 

## MMCA_with_vaccination
Piergiorgio Castioni extended the original MMCAcovid19 package to allow modelling of the effect of vaccines, herd immunity and reinfections. A new dimension was added to the model to account for three different vaccination states:
- Unvaccinated agents
- Vaccinated agents
- Residual vaccination

For agents that have received the vaccine the transition probabilities related to the most negative aspects of the disease, such as transmission, hospitalization and death are updated. In addition, the model has an additional transition rate to account for reinfections. This feature was absent in the MMCACovid19 because this package was developed in the early stage of the COVID-19 pandemic when there was no available information on reinfections nor on the emergence of new strains.

The source of the initial version can be found in the following link: MMCA_with_vaccination [https://github.com/PGcastioni/MMCA_with_vaccination/]([url](https://github.com/PGcastioni/MMCA_with_vaccination/))


## References
1. Jesús Gómez-Gardeñes, David Soriano-Paños and Alex Arenas: Critical regimes driven by recurrent mobility patterns of reaction-diffusion processes in networks, _Nature Physics_ **14** (2018) 391–395.
  
3. Alex Arenas, Wesley Cota, Jesús Gómez-Gardeñes, Sergio Gómez, Clara Granell, Joan T. Matamalas, David Soriano-Paños and Benjamin Steinegger: Modeling the spatiotemporal epidemic spreading of COVID-19 and the impact of mobility and social distancing interventions, _Physical Review X_ **10** (2020) 041055.

4. Castioni, P., Gómez, S., Granell, C., & Arenas, A. (2024). Rebound in epidemic control: How misaligned vaccination timing amplifies infection peaks. npj Complexity, 1(1), 20.

