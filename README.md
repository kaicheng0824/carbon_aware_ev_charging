# Carbon Aware EV Charging
Downloading 2021 data from CAISO regarding carbon emission and power supply data to compute the carbon intensity of grid for EV charging optimization. 

## Files
### Dataset (CAISO Emission data)
- Emission Data: http://www.caiso.com/todaysoutlook/pages/emissions.html
    - Zip File in this Repo: [CAISO_CO2_per_resource_2021.zip](data/CO2/CAISO_CO2_per_resource_2021.zip)
- Supply Data: https://www.caiso.com/todaysoutlook/Pages/supply.html
    - Zip File in this Repo: [CAISO_supply_2021.zip](data/Supply/CAISO_supply_2021.zip)

### Consolidate python script
These 2 scripts uses the csv library to iterate through all dates to summarize the year round data in terms of average carbon intensity.
- [consolidate_co2.py](./consolidate_co2.py): Consolidate emission data throughout the year of 2021
- [consolidate_supply.py](./consolidate_supply.py): Consolidate energy supply data throughout the year of 2021

### Visualization
- https://docs.google.com/spreadsheets/d/1brIf13YIAzqOG9SNOqqScco_z9aQ8DBt6TafIYevyAI/edit?usp=sharing
