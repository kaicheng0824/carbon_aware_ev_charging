# Carbon Aware EV Charging
Carbon-aware EV charging is a collaborative project between the Lawrence Berkeley National Laboratory and the Department of Electrical and Computer Engineering with the goal of discovering an EV charging strategy that minimizes carbon emission during EV charging sessions. 

We use dataset from the California Independent System Operator as well as the EV charging session data from the Berkley lab to simulate the optimization problem.

## Files
### Dataset (CAISO Emission data)
- Emission Data: http://www.caiso.com/todaysoutlook/pages/emissions.html
    - Zip File in this Repo: [CAISO_CO2_per_resource_2021.zip](data/CO2/CAISO_CO2_per_resource_2021.zip)
- Supply Data: https://www.caiso.com/todaysoutlook/Pages/supply.html
    - Zip File in this Repo: [CAISO_supply_2021.zip](data/Supply/CAISO_supply_2021.zip)
- EV Charging Dataset (Provided by Berkley Lab)
    - [LBNL_Data.csv](data/Berkley_EV_Charging/LBNL_Data.csv)

### Consolidate python script
These 2 scripts uses the csv library to iterate through all dates to summarize the year round data in terms of average carbon intensity.
- [consolidate_co2.py](./consolidate_co2.py): Consolidate emission data throughout the year of 2021
- [consolidate_supply.py](./consolidate_supply.py): Consolidate energy supply data throughout the year of 2021

### Visualization
Carbon Intensity Visualization
- https://docs.google.com/spreadsheets/d/1brIf13YIAzqOG9SNOqqScco_z9aQ8DBt6TafIYevyAI/edit?usp=sharing


# Use
Install packages
```python
pip install requirements.txt
```
Generating results
```python
cd optimization
python TOU_2021.py --P 120
python EDF_2021.py --P 120
python ES_2021.py --P 120
python carbon_aware.py --P 120 --factor 0.35
```

Go to ```result/result_analysis.ipynb``` to analysis the results.
