import copy
import numpy as np
from prettytable import PrettyTable
import subsets
import pandas as pd
import matplotlib.pyplot as plt
import constant
from ihe_model import IHE_model
from mrs_model import mRS_Model 


def get_hemo_stats(data):
    a = copy.deepcopy(data)
    b = data.loc[data.xa_inhib == 1, :].copy() #doac
    c =  data.loc[data.anticoag_home___1 == 1, :].copy() #warfarin
    d = data.loc[np.array((data.anticoag_home___1 == 0) & (data.xa_inhib == 0)), :].copy() #neither

    a.loc[:, 'expansion_percentage'] = a.loc[:, 'bleed_exp_vol'] / a.loc[:, 'ich_volume']
    b.loc[:, 'expansion_percentage'] = b.loc[:, 'bleed_exp_vol'] / b.loc[:, 'ich_volume']
    c.loc[:, 'expansion_percentage'] = c.loc[:, 'bleed_exp_vol'] / c.loc[:, 'ich_volume']
    d.loc[:, 'expansion_percentage'] = d.loc[:, 'bleed_exp_vol'] / d.loc[:, 'ich_volume']

    a.loc[:, 'expansion_percentage'] = a.loc[:, 'expansion_percentage'].fillna(0)
    b.loc[:, 'expansion_percentage'] = b.loc[:, 'expansion_percentage'].fillna(0)
    c.loc[:, 'expansion_percentage'] = c.loc[:, 'expansion_percentage'].fillna(0)
    d.loc[:, 'expansion_percentage'] = d.loc[:, 'expansion_percentage'].fillna(0)

    a.loc[:, 'poor_hemo'] = np.where(a.loc[:, 'expansion_percentage'] > 0.35, 1, 0)
    b.loc[:, 'poor_hemo'] = np.where(b.loc[:, 'expansion_percentage'] > 0.35, 1, 0)
    c.loc[:, 'poor_hemo'] = np.where(c.loc[:, 'expansion_percentage'] > 0.35, 1, 0)
    d.loc[:, 'poor_hemo'] = np.where(d.loc[:, 'expansion_percentage'] > 0.35, 1, 0)

    a.loc[:, 'good_hemo'] = np.where((a.loc[:, 'expansion_percentage'] > 0.2) & (a.loc[:, 'expansion_percentage'] <= 0.35), 1, 0)
    b.loc[:, 'good_hemo'] = np.where((b.loc[:, 'expansion_percentage'] > 0.2) & (b.loc[:, 'expansion_percentage'] <= 0.35), 1, 0)
    c.loc[:, 'good_hemo'] = np.where((c.loc[:, 'expansion_percentage'] > 0.2) & (c.loc[:, 'expansion_percentage'] <= 0.35), 1, 0)
    d.loc[:, 'good_hemo'] = np.where((d.loc[:, 'expansion_percentage'] > 0.2) & (d.loc[:, 'expansion_percentage'] <= 0.35), 1, 0)

    a.loc[:, 'excel_hemo'] = np.where(a.loc[:, 'expansion_percentage'] <= 0.2, 1, 0)
    b.loc[:, 'excel_hemo'] = np.where(b.loc[:, 'expansion_percentage'] <= 0.2, 1, 0)
    c.loc[:, 'excel_hemo'] = np.where(c.loc[:, 'expansion_percentage'] <= 0.2, 1, 0)
    d.loc[:, 'excel_hemo'] = np.where(d.loc[:, 'expansion_percentage'] <= 0.2, 1, 0)

    anticoagulant_table = PrettyTable()

    anticoagulant_table.field_names = ['Anticoagulant Status', 'Bleed Expansion Rate', 'Excellent HSE (%)', 'Good HSE (%)', 'Poor HSE(%)', 'total_n']

    def add_anticoagulant(table, status, rate, e, g, p, tn):
        table.add_row([status, round(rate, 3), round(e * 100, 1), round(g*100, 1), round(p*100, 1), tn])

    #     print(np.mean(a.excel_hemo) + np.mean(a.good_hemo) + np.mean(a.poor_hemo))
    add_anticoagulant(anticoagulant_table,
                      "All Patients",
                      np.mean(a.yn_expansion),
                      np.mean(a.excel_hemo),
                      np.mean(a.good_hemo),
                      np.mean(a.poor_hemo),
                     len(a))
    add_anticoagulant(anticoagulant_table,
                      "DOAC Patients",
                      np.mean(b.yn_expansion),
                      np.mean(b.excel_hemo),
                      np.mean(b.good_hemo),
                      np.mean(b.poor_hemo),
                     len(b))
    add_anticoagulant(anticoagulant_table,
                      "Warfarin Patients",
                      np.mean(c.yn_expansion),
                      np.mean(c.excel_hemo),
                      np.mean(c.good_hemo),
                      np.mean(c.poor_hemo),
                     len(c))
    add_anticoagulant(anticoagulant_table,
                      "Else Patients (not Warfarin, not DOAC)",
                      np.mean(d.yn_expansion),
                      np.mean(d.excel_hemo),
                      np.mean(d.good_hemo),
                      np.mean(d.poor_hemo),
                     len(d))
    print(anticoagulant_table)

def generate_simulated_DOAC_cohort(real_world_cohort):
    sim_population = copy.deepcopy(real_world_cohort)

    sim_population.loc[:, 'anticoag_home___1'] = 0
    sim_population.loc[:, 'other_coag'] = 0
    sim_population.loc[:, 'xa_inhib'] = 1

    return sim_population

def create_average_patient_dataset(dataset, features_of_interest, iterating_feature):
    ns = 100
    mrs_features_wo_ihe_prop = copy.deepcopy(features_of_interest)
    mrs_features_wo_ihe_prop.remove(iterating_feature)

    mean_feature_vals = {
        x: np.mean(dataset[x]) for x in mrs_features_wo_ihe_prop
    }

    mean_feature_vals[iterating_feature] = [i/ns for i in range(ns+1)]

    mean_dataset = pd.DataFrame(data = mean_feature_vals,
                               columns = mrs_features_wo_ihe_prop + [iterating_feature])
    return mean_dataset

def get_andexanet_effects(dataset, ihe_model, mrs_model, ns):
    andexanet_percent_reduction = [i/ns for i in range(ns+1)]

    administered_population = copy.deepcopy(dataset)

    #admister the drug
    original_ihe_propensity = ihe_model.predict_proba(administered_population[ihe_model.ihe_feature_names])[:, 1]
    administered_population.loc[:, 'ihe_propensity'] = original_ihe_propensity

    all_mean_simulated_poor_mRS_propensity = []

    for percent_reduction in andexanet_percent_reduction:
        administered_population.loc[:, 'ihe_propensity'] = original_ihe_propensity * (1 - percent_reduction)

        simulated_poor_mRS_propensity = mrs_model.predict_proba(administered_population[mrs_model.mrs_feature_names])[:, 1]
        mean_simulated_poor_mRS_propensity = np.mean(simulated_poor_mRS_propensity)

        all_mean_simulated_poor_mRS_propensity.append(mean_simulated_poor_mRS_propensity)
    
    reduction_as_percentage = np.array(andexanet_percent_reduction) * 100
    
    return reduction_as_percentage, all_mean_simulated_poor_mRS_propensity 

def get_mrs_effect_plots(population, population_name, ihe_model, mrs_model, ns): 
    reduction_as_percentage, all_mean_simulated_poor_mRS_propensity = get_andexanet_effects(population, ihe_model, mrs_model, ns)

    plt.xlabel("% Reduction in IHE Patients by Andexanet")
    plt.ylabel("Mean Propensity for Poor mRS Outcome")
    plt.title(f"Effect of Andexanet on Mean Propensity for Poor mRS Outcome\n(Sample: {population_name})\nSample Size: {len(population)}")
    plt.plot(reduction_as_percentage, all_mean_simulated_poor_mRS_propensity)

def get_nnt_and_cost_effects(dataset, ihe_model, mrs_model, ns):    
    reduction_as_percentage, simulated_poor_mRS__probs = get_andexanet_effects(dataset, ihe_model, mrs_model, ns)
    no_drug_poor_mRS_prob = simulated_poor_mRS__probs[0]
    simulated_poor_mRS__probs[0] = np.nan
    
    nnt = 1/(no_drug_poor_mRS_prob - np.array(simulated_poor_mRS__probs))
    thousand_to_treat_one = nnt * constant.COST_THOUSANDS
    
    return reduction_as_percentage, nnt, thousand_to_treat_one

def get_NNT_cost_plots(dataset, cohort_name, ihe_model, mrs_model, ns):
    reduction_as_percentage, nnt, thousand_to_treat_one = get_nnt_and_cost_effects(
        dataset,
        ihe_model, 
        mrs_model,
        ns)    

    plt.figure(figsize = (15,5))
    plt.subplot(121)

    plt.xlabel("% Reduction in IHE Patients by Andexanet")
    plt.ylabel("NNT")
    plt.ylim((0, 100))
    plt.title(f"NNT\n(Sample: {cohort_name} | n = {len(dataset)})")
    plt.plot(reduction_as_percentage, nnt)

    plt.subplot(122)
    plt.xlabel("% Reduction in IHE Patients by Andexanet")
    plt.ylabel("USD (thousand)")
    plt.ylim((0, 100 * constant.COST_THOUSANDS))
    plt.title(f"Cumulative Treatment Costs to Prevent One Unfavorable Outcome\n(Sample: {cohort_name} | n = {len(dataset)})")
    plt.plot(reduction_as_percentage, thousand_to_treat_one)

    plt.tight_layout()
    print("NNT at 33% Reduction: ", nnt[33])
    print("NNT at 50% Reduction: ", nnt[50])
    print("NNT at 100% Reduction: ", nnt[-1])
    plt.show()

def setup_ihe_alternatives(ihe_model, ihe_df, ihe_feature_names):
    ihe_model_container = {
        'name': 'Final Model...',
        'description': 'IHE model to run simulation and analysis',
        'feature_names': ihe_feature_names,
        'model': ihe_model,
        'CV-AUROC': '0.783'
    }

    #alternative A
    ihe_alternative_A = {
        'name': 'Alternative 1..',
        'description': 'Same features as optimal but training does not incorporate ProWSyn',
        'feature_names': ihe_feature_names
    }
    ihe_alternative_A['model'] = IHE_model(
        ihe_alternative_A['feature_names'],
        'inefficient_hemo',
        ihe_df,
        smote = False
    )
    ihe_alternative_A['CV-AUROC'] = str(np.round(ihe_alternative_A['model'].get_metrics(ihe_df).Values[0], 3))

    #alternative B
    b_features = ihe_feature_names.copy()
    b_features.remove('soc___1')
    ihe_alternative_B = {
        'name': 'Alternative 2..',
        'description': 'Optimal Model without Alcohol Abuse feature',
        'feature_names': b_features
    }
    ihe_alternative_B['model'] = IHE_model(
        ihe_alternative_B['feature_names'],
        'inefficient_hemo',
        ihe_df,
        smote = True
    )
    ihe_alternative_B['CV-AUROC'] = str(np.round(ihe_alternative_B['model'].get_metrics(ihe_df).Values[0], 3))

    #alternative C
    ihe_alternative_C = {
        'name': 'Alternative 3..',
        'description': 'Same as the IHE model except initial ICH volume and LKW_hct1 are not log-transformed',
        'feature_names': ['ich_volume', 'soc___1', 'lkw_hct1', 'anticoag_home___1', 'xa_inhib', 'ich_spotsign', 'other_coag']
    }
    ihe_alternative_C['model'] = IHE_model(
        ihe_alternative_C['feature_names'],
        'inefficient_hemo',
        ihe_df,
        smote = True
    )
    ihe_alternative_C['CV-AUROC'] = str(np.round(ihe_alternative_C['model'].get_metrics(ihe_df).Values[0], 3))

    #alternative D
    ihe_alternative_D = {
        'name': 'Alternative 4..',
        'description': 'Model with only anticoagulation status',
        'feature_names': ['anticoag_home___1', 'xa_inhib', 'other_coag']
    }
    ihe_alternative_D['model'] = IHE_model(
        ihe_alternative_D['feature_names'],
        'inefficient_hemo',
        ihe_df,
        smote = True
    )
    ihe_alternative_D['CV-AUROC'] = str(np.round(ihe_alternative_D['model'].get_metrics(ihe_df).Values[0], 3))

    return [ihe_alternative_A, ihe_alternative_B, ihe_alternative_C, ihe_alternative_D, ihe_model_container]

def setup_mrs_alternatives(mrs_model, train_mrs_df, mrs_feature_names, bin_mrs_outcome_name, no_smote_ihe_model):
    mrs_container = {
        'name': 'Final Model...',
        'description': 'Outcome Model used for analysis',
        'feature_names': mrs_feature_names,
        'model': mrs_model,
        'CV-AUROC': '0.816',
        'test_df': train_mrs_df
    }

    #alternative A
    mrs_alternative_A = {
        'name': 'Alternative 1..',
        'description': 'IHE propensity as only feature',
        'feature_names': ['ihe_propensity'],
        'test_df': train_mrs_df
    }
    mrs_alternative_A['model'] = mRS_Model(
        mrs_alternative_A['feature_names'],
        bin_mrs_outcome_name,
        train_mrs_df
    )
    mrs_alternative_A['CV-AUROC'] = str(np.round(mrs_alternative_A['model'].get_AUROCC_AUPRC_metrics().Values[0], 3))

    return [mrs_alternative_A, mrs_container]