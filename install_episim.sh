#!/bin/bash
cd model
git clone https://github.com/Epi-Sim/EpiSim.jl.git
cd EpiSim.jl/
bash install.sh
mv episim ../
cd ..
cd ..

