import numpy as np
import pandas as pd

def get_ANNEXA4_comparable_cohort(df, feature_dictionary, verbose = True):
    """
    Get the data of the patients who match the inclusion criteria from the
    ANNEXA-4 study. Removes all patients with initial ICH volumes larger
    than 60cc, low GCS, time between LKW and first head CT over 18 hours,
    and early WLST.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe comprising the entire data
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    verbose : Boolean, optional
        Printsthe number of patients excluded with each inclusion criteria

    Returns
    -------
    pandas.DataFrame
        Data of patients who match the inclusion criteria from the ANNEXA-4
        study
    """
    b = len(df)
    included = df[df[feature_dictionary["Initial ICH Volume"]] <= 60].copy()
    c = len(included)
    included = included[included[feature_dictionary["Initial GCS Score: 3-4"]] == 0].copy()
    d = len(included)
    included = included[included[feature_dictionary["Hours from LKW to hospital arrival"]] < 18].copy()
    e = len(included)
    included = included[included[feature_dictionary["CMO/WLST at admission"]] == 0].copy()
    f = len(included)

    if verbose:
        print("Generating the ANNEXA-4-comparable Cohort...")
        print("\tInitial ICH Volume >60cc:", b - c, "removed")
        print("\tInitial GCS 3-4:", c -d, "removed")
        print("\tTime between LKW and first head CT >18 hrs:", d-e, "removed")
        print("\tCMO/WLST status at admission:", e-f, "removed")
        print("\tLength before Exclusion:", len(df))
        print("\tLength after Exclusion:", len(included))
        print()
    
    included.reset_index(inplace = True, drop = True)
    return included

def get_ANNEXA4_ineligible_cohort(df, feature_dictionary):
    """
    Get the data of the patients who do not match the inclusion criteria 
    from the ANNEXA-4 study.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe comprising the entire data
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file

    Returns
    -------
    pandas.DataFrame
        Data of patients who do not match the inclusion criteria from the
        ANNEXA-4 study
    """
    included_study_id = get_ANNEXA4_comparable_cohort(df, feature_dictionary, verbose = False)[feature_dictionary["Study ID"]]
    
    study_ids = df[feature_dictionary["Study ID"]]
    excluded = df[np.invert(study_ids.isin(included_study_id))].copy()
    
    excluded.reset_index(inplace = True, drop = True)
    return excluded

def get_FXai_df(df, feature_dictionary):
    """
    Get the data of the patients who are on Faxtor Xa inhibitors

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe comprising the entire data
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file

    Returns
    -------
    pandas.DataFrame
        Data of patients who are on Faxtor Xa inhibitors
    """
    x = df.loc[np.where(df[feature_dictionary["FXa inhibitor use"]] == 1, True, False)].copy()
    x.reset_index(inplace= True)
    return x

def get_higher_likelihood_of_favorable_outcome_in_ANNEXA4_cohort(df, feature_dictionary, verbose = True):
    """
    Get the data of the ANNEXA-4-comparable patients who have a higher
    likelihood of favorable function outcome. First, the ANNEXA-4-comparable
    cohort is generated. Then patients who were WLST status at any time of
    hospital stay, were discharged to hospice, dead at discharge (mRS = 6),
    or had initial GCS scores not between 13-15 are removed.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe comprising the entire data
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file

    Returns
    -------
    pandas.DataFrame
        Data of of the ANNEXA-4-comparable patients who have a higher
        likelihood of favorable function outcome
    """
    included_df = get_ANNEXA4_comparable_cohort(df, feature_dictionary, verbose = False)

    a = len(included_df)
    included = included_df[included_df[feature_dictionary["CMO/WLST at any time of hospital stay"]] == 0].copy()
    b = len(included)
    included = included[included[feature_dictionary["Discharged to Hospice"]] == 0].copy()
    c = len(included)
    included = included[included[feature_dictionary["mRS at Discharge"]] != 6].copy()
    d = len(included)
    included = included[included[feature_dictionary["Initial GCS Score: 13-15"]] == 1].copy()
    e = len(included)

    if verbose:
        print("Getting patients with a higher likelihood of a favorable functional outcome")
        print("among the ANNEXA-4 comparable cohort...")
        print("\tCMO/WLST status at any time in hospital stay:", a-b, "removed")
        print("\tDischarged to Hospice:", b - c, "removed")
        print("\tDead at Discharge (mRS = 6):", c -d, "removed")
        print("\tInitial GCS Score not 13-15:", d-e, "removed")
        print()
    
    included.reset_index(inplace = True, drop = True)
    return included
