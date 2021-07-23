import pandas as pd 
import numpy as np 
import copy

def simulate_FXai_use(real_world_cohort, feature_dictionary):
    """
    Get a dataset where the patients have Factor Xa inhibitor (FXai) use
    simulated. To do this, warfarin use and other coagulopathy are set to False.
    FXai use is set to True. Original data is not affected.

    Parameters
    ----------
    real_world_cohort : pandas.DataFrame
        Data without FXai status simulated
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    
    Returns
    -------
    pandas.DataFrame
        Data with FXai status simulated.
    """
    sim_population = copy.deepcopy(real_world_cohort)

    sim_population.loc[:, feature_dictionary["Warfarin Use"]] = 0
    sim_population.loc[:, feature_dictionary["Other Coagulopathy"]] = 0
    sim_population.loc[:, feature_dictionary["FXa inhibitor use"]] = 1

    return sim_population

def reduce_IHE_probability_and_get_subsequent_outcome_probability(dataset, IHE_model, outcome_model):
    """
    Get the mean outcome probability if probability for IHE were to be decreased
    across a range of different probability reductions. 0% indicates no 
    probability reduction or no treatment effect. 100% indicates complete 
    probability reduction or complete treatment effect.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Dataset of patients to observe outcomes
    IHE_model : IHE_Model
        Predictive trained model for IHE
    outcome_model : Outcome_Model
        Predictive trained model for clinical poor outcome

    Returns
    -------
    numpy.array
        The percentages at which IHE probability was reduced
    numpy.array
        The mean probability for clinical outcome at each respective
        percentage reduction
    """
    #amount to reduce IHE probability by. 0 indicates no IHE probability
    #reduction. 100 indicates complete IHE probability reduction
    IHE_probability_reduction = [i/100 for i in range(100+1)]

    reduced_IHE_probability_patients = copy.deepcopy(dataset)

    #get the original probability of IHE and add to data
    original_ihe_propensity = IHE_model.predict_proba(reduced_IHE_probability_patients)[:, 1]
    reduced_IHE_probability_patients.loc[:, outcome_model.ihe_probability_feature] = original_ihe_propensity.copy()


    #get the mean outcome probability after reducing probability for IHE at
    #various percentages
    mean_outcome_probability_at_all_IHE_probability_reductions = []

    for percent_reduction in IHE_probability_reduction:
        reduced_IHE_probability_patients.loc[:, outcome_model.ihe_probability_feature] = original_ihe_propensity * (1 - percent_reduction)

        outcome_probability_after_IHE_probability_reduction = outcome_model.predict_proba(reduced_IHE_probability_patients)[:, 1]
        mean_outcome_probability_after_IHE_probability_reduction = np.mean(outcome_probability_after_IHE_probability_reduction)

        mean_outcome_probability_at_all_IHE_probability_reductions.append(mean_outcome_probability_after_IHE_probability_reduction)
    
    reduction_as_percentage = np.array(IHE_probability_reduction) * 100
    mean_outcome_probability_at_all_IHE_probability_reductions = np.array(mean_outcome_probability_at_all_IHE_probability_reductions)
    
    return reduction_as_percentage, mean_outcome_probability_at_all_IHE_probability_reductions 

def translate_outcome_probability_to_arr_nnt(mean_outcome_probabilties, round_NNT_up = True):
    """
    Get the absolute risk reduction (ARR) and numbers needed to treat (NNT) 
    given the mean outcome probabilities after reducing IHE probability scores.

    Parameters
    ----------
    mean_outcome_probabilities : numpy.array
        The mean probability for clinical outcome at each respective
        percentage reduction
    round_NNT_up : Boolean, optional
        Whether to round up the NNT values to the nearest integer. Technically,
        NNT must be integers because it represents humans. However, for when
        plotting NNT values, the non-rounded values may be useful for plot
        smoothness
    
    Returns
    np.array
        The ARR at each IHE probability reduction
    np.array
        The NNT at each IHE probability reduction
    """
    #get the control group or no probability reduction
    control_outcome_probability = mean_outcome_probabilties[0]

    #get the ARR
    arr = control_outcome_probability - mean_outcome_probabilties

    #get the NNT
    arr_for_nnt_calc = arr.copy()
    arr_for_nnt_calc[0] = np.nan
    arr_for_nnt_calc = np.array(arr_for_nnt_calc)
    nnt = 1/arr_for_nnt_calc
    
    if round_NNT_up:
        nnt = np.ceil(nnt)
    
    return arr, nnt

