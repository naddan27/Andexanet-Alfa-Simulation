# Andexanet Alfa Simulation
Andexanet alfa (Andexxa) is a biological decoy that has been approved for the acute reversal of FXa inhibitors in FXa inhibitor associated intracerebral hemorrhage (ICH) following the results of the ANNEXA-4 study (1). However, with the study being single-arm, the impact of its use in FXa inhibitor-associated ICH is unknown. Here, we present a simulation study where the effects of the drug can be estimated on own institutional data. Until data from a randomized control trial is available for andexanet alfa, we hope this study can be used to guide physicians and hospitals on effects to expect and whether to include the drug in the hospital formulary.

## Setup
### Code
Our code was written in Python 3. After cloning the repository, please install the dependencies in the *requirements.txt* file by running
```
pip install -r requirements.txt -v
```
### Dataset
To configure your institutional dataset for our code, please refer to the *variables.json* file to see which features are needed. All feature values must be saved in numerical format (True/False values can be saved as 1/0, respectively). Please save your dataset as a .csv file. We have a provided a synthetic dataset to use as reference when formating your dataset. This dataset was generated by creating random values based on the distributions from real-world data. As this is randomly generated, the results and analysis on the synthetic dataset do not carry any meaning and is solely for reference use.

## Simulation
### Predictive Model Generator
First, the code will create a logistic regression predictive model for inadaquate hemostasis, defined as >35% increase in ICH volume. Using this model, inadaquate hemostasis probability scores will be assigned for each patient. Then a logistic regression predictive model for poor clinical outcome, defined as mRS 4-6 after 3 months, will be created. The inputs for the inadaquate hemostasis model are known clinical predictors of hematoma expansion. The inputs for the poor clinical outcome model are known predictors for unfavorable clinical outcome, along with the inadaquate hemostasis probability score. These models are generated using all of the data available.

### FXa inhibitor use simulation
Our code can simulate FXa inhibitor use by setting anticoagulation status to FXa inhibitor use only.

### Andexanet Simulation
As one of the primary outcomes in the ANNEXA-4 study was the percentage of patients with inadaquate hemostasis, andexanet alfa administration is simulated by reducing the probability score generated by the inadaquate hemostasis predictive model. Subsequent clinical outcomes with andexanet alfa administration is modeled by feeding the reduced probability score into the poor outcome model. Based on the differences in unfavorable outcome probabilities between the no probability score reduction group (control) and the reduced probabiltiy score group (treatment group), the absolute risk reduction, number needed to treat, and cumulative cost to prevent one unfavorable outcome are calculated. With only a single-arm in the ANNEXA-4 study, the exact reduction in probability score expected by andexanet alfa is unknown. Therefore, this simulation reports results at all probability reductions from 0-100%. However, 33% and 50% probability score reductions are shown explicitly, as these thresholds represent optimistic effects that would be expected to have a clinically relevant impact on outcomes.

## Running the Code
The simulation code is saved in a Jupyter Notebook for ease of use. Please refer to *Andexanet Alfa Simulation.ipynb*. Here, a simulation run on the simulation run is saved for reference.

## References
1. Connolly SJ, Crowther M, Eikelboom JW, et al. Full Study Report of Andexanet Alfa for Bleeding Associated with Factor Xa Inhibitors. New England Journal of Medicine. 2019;380(14):1326-1335. doi:10.1056/NEJMoa1814051