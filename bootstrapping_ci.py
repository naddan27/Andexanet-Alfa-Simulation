import numpy as np
import pandas as pd
import random
from predictive_models import *
from run_simulation import *
from tqdm import tqdm
from pqdm.processes import pqdm
from functools import partial
import json 
import psutil

def simulate_andexanet_effects(n_bootstrapped_samples,
                            training_ihe_dataset,
                            training_outcome_dataset,
                            simulation_based_cohorts,
                            simulation_based_cohort_names,
                            feature_dictionary,
                            generate_CI_by_using_bootstrapped_models_in_simulation,
                            random_state = 0,
                            parallel = True
                            ):
    """
    Simulates andexanet effects by first assigning a IHE probability score. Then,
    this probability score is reduced from 0-100% for each patient. Using the
    reduced probability score, a new poor outcome probability is generated.
    The ARR and NNT is then derived from the difference between 0% probability
    reduction, which would represent no treatment effect, and x% probability
    reduction. To generate confidence intervals, bootstrapping is performed.

    Parameters
    ----------
    n_bootstrapped_samples : int
        Number of samples to bootstrap in generating the confidence intervals
    training_ihe_dataset : pandas.DataFrame
        Dataset to train the IHE model
    training_outcome_dataset : pandas.DataFrame
        Dataset to train the Outcome model
    simulation_based_cohorts: list of pandas.DataFrame
        Each element represents the data on which to perform the simulation on.
        E.g. the ANNEXA-4-comparable cohort
    simulation_based_cohort_names : list of str
        User-assigned name of the simulation cohort
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    generate_CI_by_using_bootstrapped_models_in_simulation : Boolean
        When doing the simulation, whether to use the bootstrapped models.
        If False, use the models on the original datasets and then the 
        simulation cohort is bootstrapped instead to generate confidence
        intervals
    random_state : int, optional
        The seed to set the random bootstrap sampling and model building
    parallel : Boolean, optional
        Whether to run the algorithm in parallel processes. Note that setting
        this to true may make your computer unuseable while the algorithm is
        running
    
    Returns
    -------
    Simulation_Results
        Object holding the results of the run
    """
    #set the random seeds
    np.random.seed(random_state)
    random.seed(random_state)
    random_seeds_for_each_bootstrapping = [int(random.random()*10000) for i in range(n_bootstrapped_samples)]

    if len(simulation_based_cohorts) != len(simulation_based_cohort_names):
        raise AssertionError("The names of the simulation based cohorts do not \
            match in length of the simulation cohorts")

    #get outcome effects on full data without bootstrapping
    all_cohorts_Subsequent_mean_outcome_probabilities, all_cohorts_Subsequent_ARR, all_cohorts_Subsequent_NNT, ihe_model_coef, outcome_model_coef, ihe_model_full, outcome_model_full = run_single_bootstrap_simulation(
                training_ihe_dataset = training_ihe_dataset,
                training_outcome_dataset = training_outcome_dataset,
                simulation_based_cohorts = simulation_based_cohorts,
                feature_dictionary = feature_dictionary,
                no_bootstrap = True,
                generate_CI_by_using_bootstrapped_models_in_simulation = [generate_CI_by_using_bootstrapped_models_in_simulation, []],
                return_models=True,
                random_state = random_state
                )

    #setup data that will hold the bootstrapped values
    n_simulation_cohorts = len(simulation_based_cohorts)
    bootstrapped_All_cohorts_Subsequent_mean_outcome_probabilities = [[] for i in range(n_simulation_cohorts)]
    bootstrapped_All_cohorts_Subsequent_ARR = [[] for i in range(n_simulation_cohorts)]
    bootstrapped_All_cohorts_Subsequent_NNT = [[] for i in range(n_simulation_cohorts)]
    bootstrapped_ihe_model_coef = []
    bootstrapped_outcome_model_coef = []

    #modify run_single_bootstrap_simulation function to only accept random seed argument
    partial_run_single_bootstrap_simulation = partial(run_single_bootstrap_simulation, training_ihe_dataset, training_outcome_dataset, simulation_based_cohorts, feature_dictionary, False, [generate_CI_by_using_bootstrapped_models_in_simulation, [ihe_model_full, outcome_model_full]], False)

    if parallel:
        bootstrapped_results = pqdm(random_seeds_for_each_bootstrapping, partial_run_single_bootstrap_simulation, psutil.cpu_count(logical=False))
    else:
        bootstrapped_results = [partial_run_single_bootstrap_simulation(x) for x in tqdm(random_seeds_for_each_bootstrapping)]

    #modify the bootstrapped results organization
    for bootstrapped_result in bootstrapped_results:
        single_all_cohorts_Subsequent_mean_outcome_probabilities = bootstrapped_result[0] #shape: nSubsets x probability reductions
        single_all_cohorts_Subsequent_ARR = bootstrapped_result[1]
        single_all_cohorts_Subsequent_NNT = bootstrapped_result[2]
        single_ihe_model_coef = bootstrapped_result[3] #shape: nfeatures,
        single_outcome_model_coef = bootstrapped_result[4]

        for subset_ix in range(n_simulation_cohorts):
            bootstrapped_All_cohorts_Subsequent_mean_outcome_probabilities[subset_ix].append(single_all_cohorts_Subsequent_mean_outcome_probabilities[subset_ix])
            bootstrapped_All_cohorts_Subsequent_ARR[subset_ix].append(single_all_cohorts_Subsequent_ARR[subset_ix])
            bootstrapped_All_cohorts_Subsequent_NNT[subset_ix].append(single_all_cohorts_Subsequent_NNT[subset_ix])
        
        bootstrapped_ihe_model_coef.append(single_ihe_model_coef)
        bootstrapped_outcome_model_coef.append(single_outcome_model_coef)

    #shapes for bootstrapped_All_cohorts_Subsequent_mean_outcome_probabilities, bootstrapped_All_cohorts_Subsequent_ARR, bootstrapped_All_cohorts_Subsequent_NNT
    # list size of n_simulation_cohorts
    # each element: (n_bootstrapped_samples x 101)
    bootstrapped_All_cohorts_Subsequent_mean_outcome_probabilities = [np.array(x) for x in bootstrapped_All_cohorts_Subsequent_mean_outcome_probabilities]
    bootstrapped_All_cohorts_Subsequent_ARR = [np.array(x) for x in bootstrapped_All_cohorts_Subsequent_ARR]
    bootstrapped_All_cohorts_Subsequent_NNT = [np.array(x) for x in bootstrapped_All_cohorts_Subsequent_NNT]
    
    # shape for bootstrapped coefficients
    # (n_bootstrapped_samples x n_coefficients)
    bootstrapped_ihe_model_coef = np.array(bootstrapped_ihe_model_coef)
    bootstrapped_outcome_model_coef = np.array(bootstrapped_outcome_model_coef)

    original_data_dict = {
        "Mean Outcome Probability": all_cohorts_Subsequent_mean_outcome_probabilities,
        "ARR": all_cohorts_Subsequent_ARR,
        "NNT": all_cohorts_Subsequent_NNT,
        "IHE Model Coef": ihe_model_coef,
        "Outcome Model Coef": outcome_model_coef,
    }

    bootstrapped_data_dict = {
        "Mean Outcome Probability": bootstrapped_All_cohorts_Subsequent_mean_outcome_probabilities,
        "ARR": bootstrapped_All_cohorts_Subsequent_ARR,
        "NNT": bootstrapped_All_cohorts_Subsequent_NNT,
        "IHE Model Coef": bootstrapped_ihe_model_coef,
        "Outcome Model Coef": bootstrapped_outcome_model_coef,
    }
    
    simulation_based_cohort_lengths = [len(x) for x in simulation_based_cohorts]

    return Simulation_Results(original_data_dict, bootstrapped_data_dict, simulation_based_cohort_names, ihe_model_full, outcome_model_full, simulation_based_cohort_lengths)

class Simulation_Results:
    """
    Class to store the results of the simulation with bootstrapping

    Parameters
    ----------
    original_data_dict : dict
        Dict must have the following keys:
            "Mean Outcome Probability": list of numpy.arrays, eacn representing
                the mean probability after IHE probability reductions of a subset
                from 0-100% (shape: (101,))
            "ARR": list of numpy.arrays, each representing the ARR after IHE
                probability reductions from 0-100% of a subset (shape: (101,))
            "NNT": list of numpy.arrays, each representing the NNT after IHE
                probability reductions from 0-100% of a subset (shape: (101,))
            "IHE Model Coef": numpy.array of coefficients in the IHE model
                (shape: (number of coefficients,1))
            "Outcome Model Coef": numpy.array of coefficients in the outcome
                model (shape: (number of coefficients, 1))
    bootstrapped_data_dict : dict
        Dict must have the following keys:
            "Mean Outcome Probability": list of 2D numpy.arrays, where each 
                2D array is associated to a subset and represents the mean outcome
                probabilities at each IHE probability reduction across all bootstrapped
                samples. Each array will have a shape of 
                (nBootstrappedSamples, 101), where 101 represents each probability
                reduction
            "ARR": list of 2D numpy.arrays, where each 2D array is associated
                to a subset and represents the ARR at each IHE probability 
                reduction across all bootstrapped samples. Each array will
                have a shape of (nBootstrappedSamples, 101), where 101
                represents each probability reduction
            "NNT": list of 2D numpy.arrays, where each 2D array is associated
                to a subset and represents the NNT at each IHE probability
                reduction across all bootstrapped samples. Each array will
                have a shape of (nBootstrappedSamples, 101), where 101
                represents each probability reduction
            "IHE Model Coef": 2D numpy.array
                Each row represents a different bootstrapped sample.
            "Outcome Model Coef": 2D numpy.array
                Each row represents a different bootstrapped sample
    simulation_based_cohort_names : list
        The user-assigned names to each simulation group
    ihe_model : IHE_Model
        The IHE model on the full dataset (no bootstrapping)
    outcome_model : Outcome_Model
        The outcome model on the full dataset (no bootstrapping)
    simulation_based_cohort_lengths : list
        Array of the lengths of each simulation cohort
    
    Methods
    -------
    get_coefficient_dfs(self, round_to = None, display_df = True)
        Gets the IHE model and outcome model coefficients, odd ratios, and 
        confidence intervals.
    get_arr_at_33_50_IHE_probability_reduction(self, subset_ix)
        Get the ARR and ARR 95% CI at 33% and 50% IHE probability reduction
    get_NNT_at_33_50_IHE_probability_reduction(self, subset_ix)
        Get the NNT and NNT 95% CI at 33% and 50% IHE probability reduction
    get_cum_cost_at_33_50_IHE_probability_reduction(self, subset_ix)
        Get the cumulative cost (cum cost) for one person to go from unfavorable
        to favorable outcomes and cum cost 95% CI at 33% and 50% IHE 
        probability reduction.
    format_confident_intervals(self, xhat, lower_95_CI, upper_95_CI)
        Combines the statistic and confidence interval into a string
    get_metrics(self)
        Gets the coefficients of the IHE model and outcome model and shows the
        ARR and NNT of each subset.
    """
    def __init__(self, original_data_dict, bootstrapped_data_dict, simulation_based_cohort_names, ihe_model, outcome_model, simulation_based_cohort_lengths):
        """
        Parameters
        ----------
        original_data_dict : dict
            Dict must have the following keys:
                "Mean Outcome Probability": list of numpy.arrays, eacn representing
                    the mean probability after IHE probability reductions of a subset
                    from 0-100% (shape: (101,))
                "ARR": list of numpy.arrays, each representing the ARR after IHE
                    probability reductions from 0-100% of a subset (shape: (101,))
                "NNT": list of numpy.arrays, each representing the NNT after IHE
                    probability reductions from 0-100% of a subset (shape: (101,))
                "IHE Model Coef": numpy.array of coefficients in the IHE model
                    (shape: (number of coefficients,1))
                "Outcome Model Coef": numpy.array of coefficients in the outcome
                    model (shape: (number of coefficients, 1))
        bootstrapped_data_dict : dict
            Dict must have the following keys:
                "Mean Outcome Probability": list of 2D numpy.arrays, where each 
                    2D array is associated to a subset and represents the mean outcome
                    probabilities at each IHE probability reduction across all bootstrapped
                    samples. Each array will have a shape of 
                    (nBootstrappedSamples, 101), where 101 represents each probability
                    reduction
                "ARR": list of 2D numpy.arrays, where each 2D array is associated
                    to a subset and represents the ARR at each IHE probability 
                    reduction across all bootstrapped samples. Each array will
                    have a shape of (nBootstrappedSamples, 101), where 101
                    represents each probability reduction
                "NNT": list of 2D numpy.arrays, where each 2D array is associated
                    to a subset and represents the NNT at each IHE probability
                    reduction across all bootstrapped samples. Each array will
                    have a shape of (nBootstrappedSamples, 101), where 101
                    represents each probability reduction
                "IHE Model Coef": 2D numpy.array
                    Each row represents a different bootstrapped sample.
                "Outcome Model Coef": 2D numpy.array
                    Each row represents a different bootstrapped sample
        simulation_based_cohort_names : list
            The user-assigned names to each simulation group
        ihe_model : IHE_Model
            The IHE model on the full dataset (no bootstrapping)
        outcome_model : Outcome_Model
            The outcome model on the full dataset (no bootstrapping)
        simulation_based_cohort_lengths : list
            Array of the lengths of each simulation cohort
        """
        self.original_data_dict = original_data_dict 
        self.bootstrapped_data_dict = bootstrapped_data_dict 
        self.simulation_based_cohort_names = simulation_based_cohort_names
        self.ihe_model = ihe_model
        self.outcome_model = outcome_model
        self.simulation_based_cohort_lengths = simulation_based_cohort_lengths

    def _get_95_percent_CI(self, list):
        q025, q975 = np.percentile(list, 2.5, axis = 0), np.percentile(list, 97.5, axis = 0)
        return q025, q975
    
    def get_coefficient_dfs(self, round_to = None, display_df = True):
        """
        Gets the IHE model and outcome model coefficients, odd ratios, and 
        confidence intervals.

        Parameters
        ----------
        round_to = int, optional
            The number of decimal points the values should be round to. If None,
            no rounding is done
        display_df = Boolean, optional
            Whether to also show the Dataframes or just return them

        Returns
        -------
        pandas.DataFrame
            The coefficients, odd ratios, and confidence intervals of the
            IHE model
        pandas.DataFrame
            The coefficients, odd ratios, and confidence intervals of the
            outcome model
        """
        #IHE model coefficients
        ihe_coef = pd.DataFrame(self.ihe_model.feature_full_name[:-1] + ["Cons"], columns = ["Feature"])
        ihe_coef["Column"] = self.ihe_model.feature_names + ["Cons"]
        ihe_coef["Coef"] = self.original_data_dict["IHE Model Coef"]
        ihe_coef["OR"] = np.exp(self.original_data_dict["IHE Model Coef"])

        lower, upper = self._get_95_percent_CI(np.exp(self.bootstrapped_data_dict["IHE Model Coef"]))
        ihe_coef["Lower 95% CI OR"] = lower
        ihe_coef["Upper 95% CI OR"] = upper

        #Outcome model coefficients
        outcome_coef = pd.DataFrame(self.outcome_model.feature_full_name[:-1] + ["Cons"], columns = ["Feature"])
        outcome_coef["Column"] = self.outcome_model.feature_names + ["Cons"]
        outcome_coef["Coef"] = self.original_data_dict["Outcome Model Coef"]
        outcome_coef["OR"] = np.exp(self.original_data_dict["Outcome Model Coef"])

        lower, upper = self._get_95_percent_CI(np.exp(self.bootstrapped_data_dict["Outcome Model Coef"]))
        outcome_coef["Lower 95% CI OR"] = lower
        outcome_coef["Upper 95% CI OR"] = upper

        #round numerical features
        coef_dfs = [ihe_coef, outcome_coef]
        numerical_features = ["Coef", "OR", "Lower 95% CI OR", "Upper 95% CI OR"]
        if round_to != None:
            for df in coef_dfs:
                for nf in numerical_features:
                    df[nf] = np.round(df[nf], 3)
        
        #display the dfs
        if display_df:
            print("IHE Model")
            display(ihe_coef)
            print()
            print("Outcome Model")
            display(outcome_coef)

        return ihe_coef, outcome_coef
    
    def get_arr_at_33_50_IHE_probability_reduction(self, subset_ix):
        """
        Get the ARR and ARR 95% CI at 33% and 50% IHE probability reduction

        Parameters
        ----------
        subset_ix : int
            The index of the subset to get the values from
        
        Returns
        -------
        tuple
            Has the ARR, lower bound, and upper bound of the ARR at 33%
            IHE probability reduction
        tuple
            Has the ARR, lower bound, and upper bound of the ARR at 50% 
            IHE probability reduction
        """
        #remember index 0 indicates 0% probability reduction
        #so probability reduction amount should match index
        arr33 = self.original_data_dict["ARR"][subset_ix][33]
        arr50 = self.original_data_dict["ARR"][subset_ix][50]

        lower, upper = self._get_95_percent_CI(self.bootstrapped_data_dict["ARR"][subset_ix])
        lowerarr33, upperarr33 = lower[33], upper[33]
        lowerarr50, upperarr50 = lower[50], upper[50]

        return (arr33, lowerarr33, upperarr33), (arr50, lowerarr50, upperarr50)
    
    def get_NNT_at_33_50_IHE_probability_reduction(self, subset_ix):
        """
        Get the NNT and NNT 95% CI at 33% and 50% IHE probability reduction

        Parameters
        ----------
        subset_ix : int
            The index of the subset to get the values from

        Returns
        -------
        tuple
            Has the NNT, lower bound, and upper bound of the NNT at 33%
            IHE probability reduction
        tuple
            Has the NNT, lower bound, and upper bound of the NNT at 50%
            IHE probability reduction
        """
        nnt33 = self.original_data_dict["NNT"][subset_ix][33]
        nnt50 = self.original_data_dict["NNT"][subset_ix][50]

        return int(np.ceil(nnt33)), int(np.ceil(nnt50))
    
    def get_cum_cost_at_33_50_IHE_probability_reduction(self, subset_ix):
        """
        Get the cumulative cost (cum cost) for one person to go from unfavorable
        to favorable outcomes and cum cost 95% CI at 33% and 50% IHE 
        probability reduction.

        Parameters
        ----------
        subset_ix : int
            The index of the subset to get the values from

        Returns
        -------
        tuple
            Has the cum cost, lower bound, and upper bound of the cum cost at 33%
            IHE probability reduction
        tuple
            Has the cum cost, lower bound, and upper bound of the cum cost at 50%
            IHE probability reduction
        """
        nnt33, nnt50 = self.get_NNT_at_33_50_IHE_probability_reduction(subset_ix)
        feature_dictionary = json.load(open("variable_names.json", mode="r"))
        cost = feature_dictionary["Andexanet alfa cost ($)"]
        return nnt33 * cost, nnt50 * cost 
    
    def format_confident_intervals(self, xhat, lower_95_CI, upper_95_CI):
        """
        Combines the statistic and confidence interval into a string

        Parameters
        ----------
        xhat : float
            The value obtained without bootstrapping
        lower_95_CI : float
            The lower bound of the 95% CI bootstrapping
        upper_95_CI : float
            The upper bound of the 05% CI bootstrapping
        
        Returns
        -------
        str
            The statistic and confidence interval formatted
        """
        xhat = np.round(xhat * 100, 1)
        lower_95_CI = np.round(lower_95_CI * 100, 1)
        upper_95_CI = np.round(upper_95_CI * 100, 1)
        
        return str(xhat) + "% (95% CI: " + str(lower_95_CI) + "%-" + str(upper_95_CI) + "%)"
    
    def get_metrics(self):
        """
        Gets the coefficients of the IHE model and outcome model and shows the
        ARR and NNT of each subset.
        """
        self.get_coefficient_dfs(round_to=3)

        #get the ARR, NNT, and cum cost at 33% and 50% IHE probability reduction
        #for each cohort
        for i in range(len(self.simulation_based_cohort_names)):
            print(self.simulation_based_cohort_names[i] + " (n=", str(self.simulation_based_cohort_lengths[i]) +  ")")
            mrs0 = self.original_data_dict["Mean Outcome Probability"][i][0]
            lower, upper = self._get_95_percent_CI(self.bootstrapped_data_dict["Mean Outcome Probability"][i])
            assert len(lower) == 101
            assert len(upper) == 101
            lowermrs0, uppermrs0 = lower[0], upper[0]
            print("\tMean Outcome Prob w/o additional treatment effect:", self.format_confident_intervals(mrs0, lowermrs0, uppermrs0))

            arr33, arr50 = self.get_arr_at_33_50_IHE_probability_reduction(i)
            print("\tARR at 33% IHE probability reduction:", self.format_confident_intervals(arr33[0], arr33[1], arr33[2]))
            print("\tARR at 50% IHE probability reduction:", self.format_confident_intervals(arr50[0], arr50[1], arr50[2]))

            nnt33, nnt50 = self.get_NNT_at_33_50_IHE_probability_reduction(i)
            print("\tNNT at 33%:", nnt33)
            print("\tNNT at 50%:", nnt50)

            cost33, cost50 = self.get_cum_cost_at_33_50_IHE_probability_reduction(i)
            print("\tCum cost at 33%: $" + str(cost33))
            print("\tCum cost at 50%: $" + str(cost50))

def sample_with_replacement(dataset, name_outcome, random_state = 0):
    """
    Sample with replacement the data to use for bootstrapping

    Parameters
    ----------
    dataset : pandas.DataFrame
        The original dataset without sampling
    name_outcome : str
        The name of the outcome feature to ensure that each class is included
        at least once. If None, then this feature is not implemented.
    random_state : int
        The seed used to fix the random sampling
    
    Returns
    -------
    pandas.DataFrame
        Sampled data after sampling with replacement
    """
    bs_data = np.zeros(dataset.shape)
    random.seed(random_state)

    positive_class_included = False
    negative_class_included = False 

    # sample up to the size of the original dataset
    for ix in range(len(dataset)):
        rand_ind = random.randint(0, len(dataset) - 1) 
        bs_data[ix, :] = dataset.loc[rand_ind, :].copy()
        
        # make sure that there is at least one positive and negative class
        if name_outcome == None:
            positive_class_included = True 
            negative_class_included = True
        elif dataset.loc[rand_ind, :][name_outcome] == 1:
            positive_class_included = True 
        elif dataset.loc[rand_ind, :][name_outcome] == 0:
            negative_class_included = True 
        else:
            raise ValueError("Dataset not properly setup in the outcome of interest name")

        # if there is neither a positive class or negative class by the
        # last sampling, replace last sampled patient with random patient
        # with missing class
        if ix == len(dataset) - 1:
            if not positive_class_included:
                subset = dataset[dataset[name_outcome] == 1].copy()
                subset.reset_index(inplace = True)
                rand_ix = random.randint(0, len(subset) - 1) 
                bs_data[ix, :] = subset.loc[rand_ix, :].copy()[dataset.columns]
            if not negative_class_included:
                subset = dataset[dataset[name_outcome] == 0]
                subset.reset_index(inplace = True)
                rand_ix = random.randint(0, len(subset) - 1) 
                bs_data[ix, :] = subset.loc[rand_ix, :].copy()[dataset.columns]

    return pd.DataFrame(data = bs_data, columns = dataset.columns.values)

def run_single_bootstrap_simulation(
                            training_ihe_dataset,
                            training_outcome_dataset,
                            simulation_based_cohorts,
                            feature_dictionary,
                            no_bootstrap,
                            generate_CI_by_using_bootstrapped_models_in_simulation,
                            return_models,
                            random_state,
                            ):
    """
    Run a simulation of andexanet administration by reducing IHE probability and
    getting the subsequent outcome probability. If bootstrapping is on, then
    the IHE model and outcome model is trained on data sampled with replacement.
    Furthermore, the cohort to implement the simulation on is also sampled.

    Parameters
    ----------
    training_ihe_dataset : pandas.DataFrame
        Data used to train the IHE Model
    training_outcome_dataset : pandas.DataFrame
        Data used to train the outcome Model
    simulation_based_cohorts : list of pandas.DataFrames
        Data to apply the andexanet simulation to and calculate changes in 
        outcome probability
    feature_dictionary : str
        feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    no_bootstrap : Boolean
        Whether to bootstrap the samples before training and simulating. Set to
        False if you want results on original data
    generate_CI_by_using_bootstrapped_models_in_simulation : [Boolean, [IHE_Model, Poor_Outcome_Model]]
        Whether to use the models trained on bootstrapped data when performing
        the simulations. If first element is False, then the simulated FXai
        cohort is bootstrapped, and the passed models are used in the simulation.
    random_state : int
        The random seed to fix random bootstrapping
    return_models : Boolean, optional
        To include the IHE and Outcome model in the returning array
    
    Returns
    list of numpy.arrays
        Each numpy.array is associated to a simulation cohort. Each numpy.array
        represents the mean probability of poor outcome after IHE probability 
        reduction across a range of reductions from 0-100%
    list of numpy.arrays
        Each numpy.array is associated to a simulation cohort. Each numpy.array
        represents the absolute risk reduction (ARR) after IHE probability 
        reduction across a range of reductions from 0-100%
    list of numpy.arrays
        Each numpy.array is associated to a simulation cohort. Each numpy.array
        represents the numbers needed to treat (NNT) after IHE probability 
        reduction across a range of reductions from 0-100%
    list
        Coefficient values of the IHE model features
    list 
        Coefficient values of the Outcome model features
    IHE_Model (dependent on 'return_models')
        IHE model used in simulation
    Outcome_Model (dependent on 'return_models')
        Outcome model used in simulation
    """
    #set the seeds
    np.random.seed(random_state)
    random.seed(random_state)

    #train IHE model with bootstrapped data
    if no_bootstrap:
        ihe_sample = training_ihe_dataset.copy()
    else:
        ihe_sample = sample_with_replacement(training_ihe_dataset, feature_dictionary["IHE"], random_state)
    ihe_model = IHE_Model(ihe_sample, smote = True, smote_proportion = 1)

    #train the outcome model with boostrapped data
    if no_bootstrap:
        mrs_sample = training_outcome_dataset.copy()
    else:
        mrs_sample = sample_with_replacement(training_outcome_dataset, feature_dictionary["Poor Outcome at 3 months"], random_state)
    mrs_sample[feature_dictionary["Probability for IHE"]] = ihe_model.predict_proba(mrs_sample)[:,1]
    outcome_model = Poor_Outcome_Model(mrs_sample)

    ihe_model_coef = ihe_model.get_coefficients()['Coefficients'].tolist()
    outcome_model_coef = outcome_model.get_coefficients()['Coefficients'].tolist()

    #create the simulated population
    all_cohorts_Subsequent_mean_outcome_probabilities = []
    all_cohorts_Subsequent_ARR = []
    all_cohorts_Subsequent_NNT = []

    #if we are not bootstrapping, use the trained model
    #if we are bootstrapping and we want to use prior model for simulation, use these
    if generate_CI_by_using_bootstrapped_models_in_simulation[0] == False and no_bootstrap == False:
        models_to_use_in_simulation = generate_CI_by_using_bootstrapped_models_in_simulation[1]
        ihe_model = models_to_use_in_simulation[0]
        outcome_model = models_to_use_in_simulation[1]

    for simulation_based_cohort in simulation_based_cohorts:
        #convert anticoagulation status to FXai use
        FXai_use_simulated_cohort = simulate_FXai_use(simulation_based_cohort, feature_dictionary)

        #sampled with replacement the FXai used simulated cohort
        if not no_bootstrap:
            if not generate_CI_by_using_bootstrapped_models_in_simulation[0]:
                FXai_use_simulated_cohort = sample_with_replacement(FXai_use_simulated_cohort, None, random_state)
        
        #get subsequent outcome probability after reducing IHE probability from
        #0-100%
        _, subsequent_mean_outcome_probabilities = reduce_IHE_probability_and_get_subsequent_outcome_probability(FXai_use_simulated_cohort, ihe_model, outcome_model)

        #get associated ARR and NNT values
        subsequent_ARRs, subsequent_NNTs = translate_outcome_probability_to_arr_nnt(subsequent_mean_outcome_probabilities, round_NNT_up=False)

        #store the arrays
        all_cohorts_Subsequent_mean_outcome_probabilities.append(subsequent_mean_outcome_probabilities)
        all_cohorts_Subsequent_ARR.append(subsequent_ARRs)
        all_cohorts_Subsequent_NNT.append(subsequent_NNTs)
    
    if return_models:
        return [all_cohorts_Subsequent_mean_outcome_probabilities,
            all_cohorts_Subsequent_ARR,
            all_cohorts_Subsequent_NNT,
            ihe_model_coef,
            outcome_model_coef,
            ihe_model,
            outcome_model
        ]
    return [all_cohorts_Subsequent_mean_outcome_probabilities,
            all_cohorts_Subsequent_ARR,
            all_cohorts_Subsequent_NNT,
            ihe_model_coef,
            outcome_model_coef
    ]