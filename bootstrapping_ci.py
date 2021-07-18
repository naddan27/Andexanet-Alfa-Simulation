from general_helper import *
import numpy as np
import pandas as pd
import random
from mrs_model import mRS_Model
from ihe_model import IHE_model
from tqdm.notebook import tqdm
from contextlib import contextmanager
import sys, os
import constant
from functools import partial
from multiprocessing import Pool
import multiprocessing

def bootstrapped_simulation(bs_num_samples,
                            training_ihe_dataset,
                            training_mrs_dataset,
                            simulation_based_cohorts,
                            simulation_based_cohort_names,
                            ihe_feature_names,
                            ihe_outcome_name,
                            mrs_feature_names,
                            mrs_outcome_name,
                            random_state,
                            parallel,
                            ihe_feature_dict,
                            mrs_feature_dict):
    random.seed(random_state)
    bs_size = (bs_num_samples, 101)

    xhat_prob, xhat_nnt, xhat_cum_cost, xhat_ihe_coef, xhat_mrs_coef, xhat_ihe_model, xhat_mrs_model = run_single_bootstrap_simulation(
                training_ihe_dataset = training_ihe_dataset,
                training_mrs_dataset = training_mrs_dataset,
                simulation_based_cohorts = simulation_based_cohorts,
                ihe_feature_names = ihe_feature_names,
                ihe_outcome_name = ihe_outcome_name,
                mrs_feature_names = mrs_feature_names,
                mrs_outcome_name = mrs_outcome_name,
                no_bootstrap = True,
                random_state = random_state)
    xhat_arr = [x ** -1 for x in xhat_nnt]

    mean_prob_bs = [[] for i in range(len(simulation_based_cohorts))]
    nnt_bs = [[] for i in range(len(simulation_based_cohorts))]
    cum_cost = [[] for i in range(len(simulation_based_cohorts))]
    ihe_coefficients = []
    mrs_coefficients = []
    arr_bs = [[] for i in range(len(simulation_based_cohorts))]

    random_seeds = [int(random.random()*10000) for i in range(bs_num_samples)]

    ic_names = [ihe_feature_dict[x] for x in ihe_feature_names] + ['Cons']
    mc_names = [mrs_feature_dict[x] for x in mrs_feature_names] + ['Cons']

    if not parallel:
        for i in tqdm(range(bs_num_samples)):
            mpb, nb, cc, ic, mc, bs_ihe_model_single, bs_mrs_model_single = run_single_bootstrap_simulation(
                training_ihe_dataset = training_ihe_dataset,
                training_mrs_dataset = training_mrs_dataset,
                simulation_based_cohorts = simulation_based_cohorts,
                ihe_feature_names = ihe_feature_names,
                ihe_outcome_name = ihe_outcome_name,
                mrs_feature_names = mrs_feature_names,
                mrs_outcome_name = mrs_outcome_name,
                no_bootstrap = False,
                random_state = random_seeds[i])

            ihe_coefficients.append(ic)
            mrs_coefficients.append(mc)

            for i in range(len(simulation_based_cohorts)):
                mean_prob_bs[i].append(mpb[i])
                nnt_bs[i].append(nb[i])
                arr_bs[i].append(nb[i]**-1)
                cum_cost[i].append(cc[i])
    else:
        partial_run_single_bootstrap_simulation = partial(run_single_bootstrap_simulation, training_ihe_dataset, training_mrs_dataset, simulation_based_cohorts, ihe_feature_names, ihe_outcome_name, mrs_feature_names, mrs_outcome_name, False)
        
        ncpus = multiprocessing.cpu_count()
        pool = Pool(processes=ncpus)
        for mean_mRS_probability, nnt, thousand_to_treat_one, ic, mc, bs_ihe_model_single, bs_mrs_model_single in pool.imap_unordered(partial_run_single_bootstrap_simulation, random_seeds):
            ihe_coefficients.append(ic)
            mrs_coefficients.append(mc)

            for i in range(len(simulation_based_cohorts)):
                mean_prob_bs[i].append(mean_mRS_probability[i])
                nnt_bs[i].append(nnt[i])
                arr_bs[i].append(nnt[i]**-1)
                cum_cost[i].append(thousand_to_treat_one[i])
    
    mean_prob_bs = [np.array(x) for x in mean_prob_bs]
    nnt_bs = [np.array(x) for x in nnt_bs]
    arr_bs = [np.array(x) for x in arr_bs]
    cum_cost = [np.array(x) for x in cum_cost]
    ihe_coefficients = np.array(ihe_coefficients)
    mrs_coefficients = np.array(mrs_coefficients)

    xhat_dict = {
        "mRS Probability": xhat_prob,
        "ARR": xhat_arr,
        "NNT": xhat_nnt,
        "Cum Cost": xhat_cum_cost,
        "IHE Coef": xhat_ihe_coef,
        "mRS Coef": xhat_mrs_coef,
    }

    bs_dict = {
        "mRS Probability": mean_prob_bs,
        "ARR": arr_bs,
        "NNT": nnt_bs,
        "Cum Cost": cum_cost,
        "IHE Coef": ihe_coefficients,
        "mRS Coef": mrs_coefficients,
    }
    
    simulation_based_cohort_lengths = [len(x) for x in simulation_based_cohorts]

    return MIT_Bootstrapped_Results(xhat_dict, bs_dict, simulation_based_cohort_names, ic_names, mc_names, xhat_ihe_model, xhat_mrs_model, simulation_based_cohort_lengths)

class MIT_Bootstrapped_Results:
    def __init__(self, xhat_dict, bs_dict, run_name, ihe_coefficient_names, mrs_coefficient_names, ihe_model, mrs_model, simulation_based_cohort_lengths):
        self.xhat_dict = xhat_dict 
        self.bs_dict = bs_dict 
        self.ihe_coefficient_names = ihe_coefficient_names 
        self.mrs_coefficient_names = mrs_coefficient_names
        self.run_name = run_name
        self.ihe_model = ihe_model
        self.mrs_model = mrs_model
        self.simulation_based_cohort_lengths = simulation_based_cohort_lengths

    def get_95_percent_CI(self, list):
        q025, q975 = np.percentile(list, 2.5, axis = 0), np.percentile(list, 97.5, axis = 0)
        return q025, q975
    
    def get_coefficient_dfs(self):
        ihe_coef = pd.DataFrame(self.ihe_coefficient_names, columns = ["Feature"])
        ihe_coef["LogReg Coef"] = self.xhat_dict["IHE Coef"]
        ihe_coef["OR"] = np.exp(self.xhat_dict["IHE Coef"])
        lower, upper = self.get_95_percent_CI(np.exp(self.bs_dict["IHE Coef"]))
        ihe_coef["Lower 95% CI OR"] = lower
        ihe_coef["Upper 95% CI OR"] = upper

        mrs_coef = pd.DataFrame(self.mrs_coefficient_names, columns = ["Feature"])
        mrs_coef["LogReg Coef"] = self.xhat_dict["mRS Coef"]
        mrs_coef["OR"] = np.exp(self.xhat_dict["mRS Coef"])
        lower, upper = self.get_95_percent_CI(np.exp(self.bs_dict["mRS Coef"]))
        mrs_coef["Lower 95% CI OR"] = lower
        mrs_coef["Upper 95% CI OR"] = upper

        return ihe_coef, mrs_coef
    
    def get_arr_33_55(self, ix):
        arr33 = self.xhat_dict["ARR"][ix][33]
        arr50 = self.xhat_dict["ARR"][ix][50]

        lower, upper = self.get_95_percent_CI(self.bs_dict["ARR"][ix])
        lowerarr33, upperarr33 = lower[33], upper[33]
        lowerarr50, upperarr50 = lower[50], upper[50]

        return (arr33, lowerarr33, upperarr33), (arr50, lowerarr50, upperarr50)
    
    def get_NNT(self, ix):
        nnt33 = self.xhat_dict["NNT"][ix][33]
        nnt50 = self.xhat_dict["NNT"][ix][50]

        return int(np.ceil(nnt33)), int(np.ceil(nnt50))
    
    def get_cum_cost(self, ix):
        nnt33, nnt50 = self.get_NNT(ix)
        cost = 24750
        return nnt33 * cost, nnt50 * cost 
    
    def arr_as_string(self, arr):
        xhat = arr[0]
        lower = arr[1]
        upper = arr[2]

        xhat = np.round(xhat * 100, 1)
        lower = np.round(lower * 100, 1)
        upper = np.round(upper * 100, 1)
        
        return str(xhat) + "% (95% CI: " + str(lower) + "%-" + str(upper) + "%)"
    
    def get_metrics(self):
        ihe_coef, mrs_coef = self.get_coefficient_dfs()
        print("IHE Model")
        display(ihe_coef)
        print("mRS Model")
        display(mrs_coef)

        for i in range(len(self.run_name)):
            print(self.run_name[i], "(n=", self.simulation_based_cohort_lengths[i], ")")
            mrs0 = self.xhat_dict["mRS Probability"][i][0]
            lower, upper = self.get_95_percent_CI(self.bs_dict["mRS Probability"][i])
            lowermrs0, uppermrs0 = lower[0], upper[0]
            print("\tmean mRS prob w/o additional treatment effect:", self.arr_as_string([mrs0, lowermrs0, uppermrs0]))

            arr33, arr50 = self.get_arr_33_55(i)
            print("\tARR at 30% IHE probability reduction:", self.arr_as_string(arr33))
            print("\tARR at 50% IHE probability reduction:", self.arr_as_string(arr50))

            nnt33, nnt50 = self.get_NNT(i)
            print("\tNNT at 30%:", nnt33)
            print("\tNNT at 50%:", nnt50)

            cost33, cost50 = self.get_cum_cost(i)
            print("\tCum cost at 30%: $" + str(cost33))
            print("\tCum cost at 50%: $" + str(cost50))

def _create_bs_sample(dataset, name_outcome, random_state = 42):
    bs_data = np.zeros(dataset.shape)
    random.seed(random_state)

    positive_class_included = False
    negative_class_included = False 

    for ix in range(len(dataset)):
        rand_ind = random.randint(0, len(dataset) - 1) 
        bs_data[ix, :] = dataset.loc[rand_ind, :].copy()
        
        if dataset.loc[rand_ind, :][name_outcome] == 1:
            positive_class_included = True 
        elif dataset.loc[rand_ind, :][name_outcome] == 0:
            negative_class_included = True 

        if ix == len(dataset) - 1:
            if not positive_class_included:
                subset = dataset[dataset[name_outcome] == 1]
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
                            training_mrs_dataset,
                            simulation_based_cohorts,
                            ihe_feature_names,
                            ihe_outcome_name,
                            mrs_feature_names,
                            mrs_outcome_name,
                            no_bootstrap,
                            random_state,
                            ):
    #generate the ihe model
    np.random.seed(random_state)
    random.seed(random_state)
    ihe_sample = _create_bs_sample(training_ihe_dataset, ihe_outcome_name, random_state)
    if no_bootstrap:
        ihe_sample = training_ihe_dataset.copy()

    ihe_model = IHE_model(ihe_feature_names, ihe_outcome_name, ihe_sample, smote = True, smote_proportion = 1)
    ic = ihe_model.get_coefficients()['Coefficients']

    #generate the mrs model
    mrs_sample = _create_bs_sample(training_mrs_dataset, mrs_outcome_name, random_state)
    if no_bootstrap:
        mrs_sample = training_mrs_dataset.copy()
    mrs_sample['ihe_propensity'] = ihe_model.predict_proba(mrs_sample[ihe_feature_names])[:,1]

    mrs_model = mRS_Model(mrs_feature_names, mrs_outcome_name, mrs_sample)
    mc = mrs_model.get_coefficients()['Coefficients']

    #create the simulated population
    mean_mRS_probability_array = []
    nnt_array = []
    thousand_to_treat_one_array = []

    for simulation_based_cohort in simulation_based_cohorts:
        sim_population = generate_simulated_DOAC_cohort(simulation_based_cohort)
        
        #apply hemostatic effects of Andexanet at 100% different reduction capacities
        _, mean_mRS_probability = get_andexanet_effects(sim_population, ihe_model, mrs_model, 100)
        mean_mRS_probability = np.array(mean_mRS_probability)
        # mean_mRS_probability = np.log(mean_mRS_probability / (1 - mean_mRS_probability))

        #translate this to NNT
        _, nnt, thousand_to_treat_one = get_nnt_and_cost_effects(
            sim_population,
            ihe_model, 
            mrs_model,
            100)  

        mean_mRS_probability_array.append(mean_mRS_probability)
        nnt_array.append(nnt)
        thousand_to_treat_one_array.append(thousand_to_treat_one)

    return mean_mRS_probability_array, nnt_array, thousand_to_treat_one_array, ic, mc, ihe_model, mrs_model

class Bootstrapped_Results:
    def __init__(self, cohort_name, mean_prob, nnt, cum_cost, ihe_coefficients, mrs_coefficients, ihe_feature_names, mrs_feature_names, ic_names, mc_names, cohort_size):
        self.cohort_name = cohort_name
        self.mean_prob = mean_prob
        self.arr = nnt**-1
        self.nnt = nnt
        self.cum_cost = cum_cost
        self.ihe_coefficients = ihe_coefficients 
        self.mrs_coefficients = mrs_coefficients
        self.ihe_feature_names = ihe_feature_names
        self.mrs_feature_names = mrs_feature_names
        self.ic_names = ic_names 
        self.mc_names = mc_names
        self.cohort_size = cohort_size

    @staticmethod
    def summary_CI_intervals(array):
        mean = np.mean(array, axis = 0)
        std = np.std(array, axis = 0)

        lower = mean - 1.96 * std 
        upper = mean + 1.96 * std 

        summary_df = pd.DataFrame()
        summary_df['Mean'] = mean
        summary_df['Lower 95% CI'] = lower 
        summary_df['Upper 95% CI'] = upper 

        return summary_df

    def summary_CI_stat_intervals(self, array):
        summary_df = self.summary_CI_intervals(array)

        summary_df.insert(loc = 0, column = 'Reduction_Capacity', value = [i for i in range(array.shape[1])])

        return summary_df

    def summary_CI_coef_intervals(self, coef_values, coef_names):
        summary_df = self.summary_CI_intervals(coef_values)

        summary_df.insert(loc = 0, column = 'Name', value = coef_names)
        summary_df["Mean OR"] = np.exp(summary_df["Mean"])
        summary_df["Lower 95% CI OR"] = np.exp(summary_df["Lower 95% CI"])
        summary_df["Upper 95% CI OR"] = np.exp(summary_df["Upper 95% CI"])
        summary_df["Log Reg Coef"] = summary_df["Mean"]
        summary_df = summary_df.drop(columns = ["Mean", "Lower 95% CI", "Upper 95% CI"])

        return summary_df


    def get_95_intervals_from_simulation(self):
        mean_prob_ci = self.summary_CI_stat_intervals(self.mean_prob)
        arr_ci = self.summary_CI_stat_intervals(self.arr)
        nnt_ci = self.summary_CI_stat_intervals(self.nnt)
        cum_cost_ci = self.summary_CI_stat_intervals(self.cum_cost)

        ihe_coef_ci = self.summary_CI_coef_intervals(self.ihe_coefficients, self.ic_names)
        mrs_coef_ci = self.summary_CI_coef_intervals(self.mrs_coefficients, self.mc_names)

        return {
            "Mean_Probability": mean_prob_ci,
            "ARR": arr_ci,
            "NNT": nnt_ci,
            "Cumulative_Cost": cum_cost_ci,
            "IHE_Coefficients": ihe_coef_ci,
            "mRS_Coefficients": mrs_coef_ci
        }