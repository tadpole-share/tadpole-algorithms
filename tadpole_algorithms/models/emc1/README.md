# TADPOLE Submission with DEBM

This repository consists of codes used by team EMC1 for the TADPOLE challenge. EMC1 were the winners of TADPOLE challenge for Ventricles prediction, and were overall runners up. 

## Required Packages and Libraries
pyebm : https://github.com/88vikram/pyebm

RStudio 1.2

Python 3.5, numpy 1.16, pandas 0.24, sklearn 0.20

## Notes

The codes have been tested in Ubuntu 18.04. Call main_EMC1.py to generate TADPOLE predictions. 

The variables tadpoleD1D2File and DEBMFolder (lines 2 and 3) should point to the directories where the TADPOLE data file and pyebm repository is placed. Edit if necessary.

Running times: Predicting the probabilities of CN, MCI, AD classes, ventricle volumes and ADAS scores takes about 3 hours. Confidence intervals for ventricles and ADAS scores (in *_bootstrap files) takes about 1 hour for each iteration. The submitted files estimates standard errors using 10 bootstrap iterations. If computed serially, it would take 10 hours, but users are strongly advised to run the bootstrap iterations in parallel if possible.
