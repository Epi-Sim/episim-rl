using Pkg

episim_src_dir = @__DIR__
episim_base_dir = dirname(episim_src_dir)
Pkg.activate(episim_base_dir)
Pkg.instantiate()

using EpiSim

EpiSim.main()
