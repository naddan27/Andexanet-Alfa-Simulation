import numpy as np 
import pandas as pd 
from collections import Counter 
from matplotlib import pyplot as plt
from general_helper import *


def get_iqr(list, roundto = 1):
    q25, q50, q75 = np.percentile(list, 25), np.percentile(list, 50), np.percentile(list, 75)
    return str(np.round(q50, roundto)) + " (" + str(np.round(q25, roundto)) + "-" + str(np.round(q75, roundto)) + ")"

def get_percentage(list):
    num_positive = Counter(list)[1]
    return str(num_positive) + " (" + str(np.round(num_positive/len(list)* 100, 1)) + "%)"

def get_dist_df(df, demographic_features, feature_dict, outcome_dict):
    outcome_name = list(outcome_dict.keys())[0]
    outcome_alt_name = outcome_dict[outcome_name]
    outcome_true_subset = df[df[outcome_name] == 1]
    outcome_false_subset = df[df[outcome_name] == 0]

    data = [demographic_features + list(feature_dict.values())]
    headers = ["Feature", outcome_alt_name + "==1 (n=" + str(len(outcome_true_subset)) + ")", outcome_alt_name + "==0 (n=" + str(len(outcome_false_subset)) + ")", "All (n=" + str(len(df)) + ")"]
    for subset in [outcome_true_subset, outcome_false_subset, df]:
        data_subset = []
            
        #add the demographic data
        for demographic_feature in demographic_features:
            if len(Counter(subset[demographic_feature])) == 2 or len(Counter(subset[demographic_feature])) == 1:
                data_subset.append(get_percentage(subset[demographic_feature]))
            else:
                data_subset.append(get_iqr(subset[demographic_feature]))

        #add the feature data
        for key in feature_dict.keys():
            #determine if the variable is categorical or continuous
            continuous = True 
            if len(Counter(subset[key])) == 2 or len(Counter(subset[key])) == 1:
                continuous = False 
            
            #if continuous, get the median and iqr
            if continuous:
                data_subset.append(get_iqr(subset[key]))
            else:
                data_subset.append(get_percentage(subset[key]))
        data.append(data_subset)

    dist_df = pd.DataFrame(data = np.transpose(data), columns = headers)
    dist_df = dist_df.style.set_properties(**{'text-align': 'left'})
    dist_df = dist_df.set_table_styles(
    [dict(selector = 'th', props=[('text-align', 'left')])])
    return dist_df

def get_95_percent_ci_for_arr(list, arr_or_nnt):
    arr_33 = list[:,33]
    arr_50 = list[:, 50]

    mean_33 = np.mean(arr_33)

def aurocc_plot(ihe_alternative_models_cum_insert, mrs_alternative_models_cum_insert):
    small_text_size = 8
    plt.figure(figsize = (3.25, 5), dpi = 600)
    plt.subplot(211)
    alt_alpha = 0.55

    ihe_alternative_models_cum = ihe_alternative_models_cum_insert.copy()
    ihe_alternative_models_cum.reverse()
    mrs_alternative_models_cum = mrs_alternative_models_cum_insert.copy()
    mrs_alternative_models_cum.reverse()

    #plot the alternative models
    colors = ['black', 'blue', 'green', 'red']
    markers = ['--', '--', '-.', '-']
    alphas = [alt_alpha, alt_alpha, alt_alpha, 1]
    for i in range(len(ihe_alternative_models_cum)):
        model_subset = ihe_alternative_models_cum[i]
        subset_name = model_subset[0]
        subset_auroc = str(round(model_subset[1]["Values"][0], 3))
        if len(subset_auroc) != 5:
            subset_auroc = subset_auroc + "0"
        subset_fpr = model_subset[2]
        subset_tpr = model_subset[3]

        plt.plot(
            subset_fpr,
            subset_tpr,
            markers[i],
            alpha = alphas[i],
            color = colors[i],
            label = subset_name + "(AUROC: " + subset_auroc + ")"
        )

    xy_line = np.arange(100)/100
    plt.plot(
        xy_line,
        xy_line,
        "-.",
        color = 'black',
        lw = 0.25,
        alpha = 1
        )

    #labels
    plt.xlabel("1-Specificity", fontsize = small_text_size)
    plt.ylabel("Sensitivity", fontsize = small_text_size)
    plt.tick_params(axis='both', which='major', labelsize= small_text_size)

    #style
    plt.style.use(['seaborn-paper'])
    plt.rc("font", family="Arial")
    plt.gca().yaxis.grid(True)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3,2,1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 4, fontsize = small_text_size,
            frameon = True, framealpha = 1)

    ######################################
    #outcome model
    plt.subplot(212)
    colors = ['blue', 'red']
    markers = ['-', '-']
    alphas = [alt_alpha, 1]
    for i in range(len(mrs_alternative_models_cum)):
        model_subset = mrs_alternative_models_cum[i]
        subset_name = model_subset[0]
        subset_auroc = str(round(model_subset[1]["Values"][0], 3))
        if len(subset_auroc) != 5:
            subset_auroc = subset_auroc + "0"
        subset_fpr = model_subset[2]
        subset_tpr = model_subset[3]

        plt.plot(
            subset_fpr,
            subset_tpr,
            markers[i],
            alpha = alphas[i],
            color = colors[i],
            label = subset_name + "(AUROC: " + subset_auroc + ")"
        )

    #plot random change auroc
    plt.plot(
        xy_line,
        xy_line,
        "-.",
        color = 'black',
        lw = 0.25,
        alpha = 1
        )

    #labels
    plt.xlabel("1-Specificity", fontsize = small_text_size)
    plt.ylabel("Sensitivity", fontsize = small_text_size)
    plt.tick_params(axis='both', which='major', labelsize= small_text_size)

    #style
    plt.style.use(['seaborn-paper'])
    plt.rc("font", family="Arial")
    plt.gca().yaxis.grid(True)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize = small_text_size, loc = 4,
            frameon = True, framealpha = 1)
    plt.tight_layout()
    # plt.subplots_adjust(left= 0.125, bottom= 0.1, right=0.95, top= 0.9, wspace= None, hspace= 0.35)
    plt.savefig("./stroke_figures/AUROC.pdf")

def unfavorable_outcome_vs_ihe_probability(mrs_df, mrs_feature_names, use_new_features_all_subsets_bs):
    sim_population = generate_simulated_DOAC_cohort(mrs_df)
    sim_included_population = subsets.get_included_df(sim_population)

    average_simulated_patient_full = create_average_patient_dataset(sim_population, mrs_feature_names, 'ihe_propensity')
    average_simulated_patient_included = create_average_patient_dataset(sim_included_population, mrs_feature_names, 'ihe_propensity')

    #plot the graph
    ns = 100
    average_simulated_patient_full_proba = use_new_features_all_subsets_bs.mrs_model.predict_proba(average_simulated_patient_full)[:,1] * 100
    average_simulated_patient_included_proba = use_new_features_all_subsets_bs.mrs_model.predict_proba(average_simulated_patient_included)[:,1] * 100

    ihe_possibilities = np.array([i/ns for i in range(ns+1)])

    small_text_size = 8

    #plot the relationship of ihe on mRS for the mean patient
    plt.figure(figsize=(3.25,2.5), dpi = 600)
    plt.plot(ihe_possibilities, average_simulated_patient_full_proba, label = "Full Cohort", alpha = 0.85)
    plt.ylabel("Probability of Unfavorable Outcome (%)", fontsize = small_text_size)
    plt.xlabel("IHE Probability Score", fontsize = small_text_size)
    plt.tick_params(axis = 'both', which = 'major', labelsize = small_text_size)

    plt.plot(ihe_possibilities, average_simulated_patient_included_proba, '-.', label = 'Comparable to ANNEXA-4 Cohort', color = (222/256, 64/256, 64/256))

    plt.legend(fontsize = small_text_size, frameon = True, framealpha = 1.0)
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rc("font", family="Arial")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("./stroke_figures/unfavorable_to_ihe_prob.pdf")

def arr_figure(use_new_features_all_subsets_bs):
    arr_included = use_new_features_all_subsets_bs.xhat_dict["ARR"][1]
    arr_excluded = use_new_features_all_subsets_bs.xhat_dict["ARR"][2]

    lower_arr_included, upper_arr_included = use_new_features_all_subsets_bs.get_95_percent_CI(use_new_features_all_subsets_bs.bs_dict["ARR"][1])
    lower_arr_excluded, upper_arr_excluded = use_new_features_all_subsets_bs.get_95_percent_CI(use_new_features_all_subsets_bs.bs_dict["ARR"][2])

    reduction_capacity = np.arange(len(arr_included))

    fig= plt.figure(figsize=(3.25,2.5), dpi = 600)
    axes= fig.add_axes([0.1,0.1,0.8,0.8])
    small_text_size = 8

    axes.fill_between(reduction_capacity,
                    upper_arr_excluded * 100,
                    lower_arr_excluded * 100,
                    facecolor = (16/256, 84/256, 13/256), #green
                    label = "ANNEXA-4 Ineligible Cohort",
                    alpha = 0.15,
                    interpolate = True,
                    linewidth = 0.5)

    axes.plot(reduction_capacity,
            arr_excluded * 100, 
            "--",
            color = (29/256, 120/256, 24/256), #green
            alpha = 1,
            linewidth = 1.75
            )

    #ANNEXA-4 Inclusion Criteria
    axes.fill_between(reduction_capacity,
                    upper_arr_included * 100,
                    lower_arr_included * 100,
                    facecolor = (158/256, 38/256, 40/256),
                    label = "ANNEXA-4 Comparable Cohort",
                    alpha = 0.20,
                    interpolate = True,
                    )

    axes.plot(reduction_capacity,
            arr_included * 100, 
            color = (179/256,18/256,18/256),
            alpha = 1,
            linewidth = 1.75
            )
    #Reorder the positioning of the items in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.grid(True, linewidth = 0.25, color = 'black', alpha = 0.5)
    legend = axes.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc = 2,
        frameon = True,
        framealpha = 1,
        prop={'size': 6}
    )

    #Set the labels
    plt.xlabel("Percent Reduction (%) in IHE Probability Score", fontsize = small_text_size)
    plt.ylabel("ARR (%)", fontsize = small_text_size)
    plt.tick_params(axis='both', which='major', labelsize= small_text_size)

    #overall style and annotations
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rc("font", family="Arial")
    plt.xlim((0,100))
    font = {'family' : 'Arial',
            'size'   : small_text_size}	
    plt.rc('font', **font)

    plt.savefig("./stroke_figures/Simulation_ARR.pdf", bbox_inches='tight')

def nnt_figure(use_new_features_all_subsets_bs):
    nnt_included = use_new_features_all_subsets_bs.xhat_dict["NNT"][1]
    nnt_excluded = use_new_features_all_subsets_bs.xhat_dict["NNT"][2]

    # lower_nnt_included, upper_nnt_included = use_new_features_all_subsets_bs.get_95_percent_CI(use_new_features_all_subsets_bs.bs_dict["NNT"][1])
    # lower_nnt_excluded, upper_nnt_excluded = use_new_features_all_subsets_bs.get_95_percent_CI(use_new_features_all_subsets_bs.bs_dict["NNT"][2])

    reduction_capacity = np.arange(len(nnt_included))

    #Simulation Study: NNT
    #ANNEXA-4 Exclusion Criteria (note that fill is purposefully before line)
    plt.figure(figsize=(3.25,5), dpi = 900)
    small_text_size = 8
    plt.subplot(211)
    plt.plot(reduction_capacity,
            nnt_excluded, 
            "--",
            color = (29/256, 120/256, 24/256), #green
            alpha = 1,
            label = "ANNEXA-4 Ineligible Cohort"
            )

    #ANNEXA-4 Inclusion Criteria
    plt.plot(reduction_capacity,
            nnt_included, 
            color = (179/256,18/256,18/256),
            label = "ANNEXA-4 Comparable Cohort",
            alpha = 1)

    #Set the labels
    plt.xlabel("Percent Reduction (%) in IHE Probability Score", fontsize = small_text_size)
    plt.ylabel("NNT", fontsize = small_text_size)
    # plt.legend()
    plt.grid(True, linewidth = 0.25, color = 'black', alpha = 0.5)
    plt.ylim((0,100))
    plt.tick_params(axis='both', which='major', labelsize= small_text_size)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon = True, framealpha = 1.0, prop={'size': 6})
    font = {'family' : 'Arial',
            'size'   : small_text_size}	
    plt.rc('font', **font)
    plt.subplot(212)
    plt.plot(reduction_capacity,
            nnt_excluded * 24.750, 
            "--",
            color = (29/256, 120/256, 24/256), #green
            alpha = 1,
            label = "ANNEXA-4 Ineligible Cohort"
            )

    #ANNEXA-4 Inclusion Criteria
    plt.plot(reduction_capacity,
            nnt_included * 24.750, 
            color = (179/256,18/256,18/256),
            label = "ANNEXA-4 Comparable Cohort",
            alpha = 1)

    #Set the labels
    plt.xlabel("Percent Reduction (%) in IHE Probability Score", fontsize = small_text_size)
    plt.ylabel("Cumulative Cost to Prevent One\nUnfavorable Outcome (thousands USD)", fontsize = small_text_size)
    # plt.legend()
    plt.grid(True, linewidth = 0.25, color = 'black', alpha = 0.5)
    plt.ylim((0,100*24.75))
    plt.tick_params(axis='both', which='major', labelsize= small_text_size)

    #overall style and annotations
    plt.style.use(['seaborn-white', 'seaborn-paper'])
    plt.rc("font", family="Arial")

    #Reorder the positioning of the items in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], frameon = True, framealpha = 1.0, prop={'size': 6})
    font = {'family' : 'Arial',
            'size'   : small_text_size}	
    plt.rc('font', **font)
    plt.tight_layout()
    plt.savefig("./stroke_figures/Simulation_NNT.pdf", bbox_inches='tight')