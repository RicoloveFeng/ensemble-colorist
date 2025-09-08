# Data Adapter for EnsembleColorist

Due to the nature of different structure from datasets, one must process their data before feeding them into EnsembleColorist.

A valid dataset should be like:
```csv
track,sample,hours,lat,lon,pressure
0,0,6,20,120,1000
...
```
where:
- `track`: the system that a data is tracking on
- `sample`: the ensemble member that giving out this data 
- `hours`: valid hour of the data
- `lat`: lantitude of the system
- `lon`: lontitude of the system
- `pressure`: center pressure of the system

We don't consider wind speed so far as different model may take different minute-average standard.

## Adapters

### FNV3Adapter

FNV3 is a AI model provided by Google. You can visit [WeatherLab](https://deepmind.google.com/science/weatherlab) and download data. 

This adapter use data under `https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble/cyclogenesis/csv`.

### AIFSAdapter

Under construction...


