FROM julia:latest

RUN apt-get update
RUN apt-get install -y git gcc apt-get install procps g++ vim python3 python3-h5netcdf python3-netcdf4


ADD episim-rl /episim-rl
WORKDIR /episim-rl/model/EpiSim.jl/
RUN sh install.sh
RUN cp episim ../../episim
WORKDIR /episim-rl/

RUN apt-get install -y python3-pandas python3-xarray python3-numpy

ENTRYPOINT ["python3", "src/myRL-episim.py", "--experiment_id", "test_1", "--config", "model/config/config_MMCACovid19.json", "--data", "model/data/", "--period", "14"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]