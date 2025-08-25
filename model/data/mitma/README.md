## Name

**COVID-19 Spain Model in MITMA areas for the MMCACovid19 engine**

## Description

This model was developed to simulate the spread of SARS-CoV-2 virus in Spain in 2020. The metapopulation is defined using 2850 mobility areas that broadly correspond to municipalities, districts or groups of the later depending on the population density.

## Provenance

The source data of this model is the 2020 mobility dataset in Spain, provided by MITMA (Ministerio de Transportes y Movilidad Sostenible).
- The zoning system (MITMA zones) is based on population density: municipalities are either aggregated or subdivided into mobility zones depending on their density.
- The population is derived from this dataset. MITMA provides detailed files with information on the number of individuals making 0, 1, 2, or more daily trips within each area. 
- To construct the mobility matrix, we selected home-to-work and work-to-home trips, which are assumed to represent the recurrent daily mobility patterns. For each pair of Mitma zones, we computed the ratio of trips from one zone to another, resulting in a normalized mobility matrix.

## Files

initial conditions:

- initial_conditions_MMCACovid19.nc/initial_conditions_MMCACovid19-vac.n: default initial conditions for both the MMCACovid19 and MMCACovid19Vac engines.

- initial_conditions-good.nc: initial conditions from 2020-02-09
- initial_conditions-med.nc: obtained from running a simulation up to 2020-03-15
- initial_conditions-bad.nc: obtained from running a simulation up to 2020-04-15 with no non-pharmaceutical interventions (NPI).

map_action.csv: maps the actions to an ID.

observables_categories: maps the observables into 5 categories (0-4).

