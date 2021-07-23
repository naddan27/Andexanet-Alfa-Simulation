from matplotlib import pyplot as plt 
import numpy as np

def get_NNT_cost_plots(xhat_NNTs, cohort_names, figsize, dpi, feature_dictionary, save_fp):
    """
    Plots the numbers needed to treat and the cumulative cost to prevent one
    unfavorable outcome vs the reduction in inadaquate hemostasis probability
    score (0-100%). Based on the numbers needed to treat passed, the cumulative
    costs are automatically calculated.

    Parameters
    ----------
    xhat_NNTs : list
        Each element corresponds to a subset. Each element comprises the number
        needed to treat for each inadaquate hemostasis probability score
        reduction
    cohort_names : list
        Each element corresponds to a str that represents the name of the subset
    figsize : tuple
        The size of the figure to generate in inches
    dpi : int
        The dots per inch to put into the figure
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    save_fp : str
        The file path to save the figure to. If None, no figure is saved
    """
    #make sure the inputs are correct
    assert len(xhat_NNTs) == len(cohort_names)
    assert len(xhat_NNTs) <= 2

    #set the parameters
    colors = ["#8c0000", "#006b37"]
    markers = ["-", "--"]

    plt.figure(figsize = figsize, dpi = dpi)

    #plot the NNTs
    plt.subplot(211)
    plt.xlabel("Percent Reduction (%) in IH Probability Score")
    plt.ylabel("NNT")
    plt.ylim((0, 100))

    reduction_percentages = np.arange(100) + 1
    for xhat_NNT, cohort_name, color, marker in zip(xhat_NNTs, cohort_names, colors, markers):
        plt.plot(reduction_percentages, xhat_NNT[1:], marker, label = cohort_name, color = color)
    
    #add legend and grid
    plt.grid(True, linewidth = 0.4, color = '#868686', alpha = 0.5)
    legend = plt.legend(frameon = True, framealpha = 1)
    frame = legend.get_frame()
    frame.set_linewidth(0.5)

    #plot the cum cost
    plt.subplot(212)
    plt.xlabel("Percent Reduction (%) in IH Probability Score")
    plt.ylabel("Cumulative Cost to Prevent One\nUnfavorable Outcome (thousands USD)")
    plt.ylim((0, 100/1000 * feature_dictionary["Andexanet alfa cost ($)"]))
    for xhat_NNT, cohort_name, color, marker in zip(xhat_NNTs, cohort_names, colors, markers):
        plt.plot(reduction_percentages, xhat_NNT[1:]/1000 * feature_dictionary["Andexanet alfa cost ($)"], marker, color = color, label = cohort_name)
    
    #add legend and grid
    plt.grid(True, linewidth = 0.4, color = '#868686', alpha = 0.5)
    legend = plt.legend(frameon = True, framealpha = 1)
    frame = legend.get_frame()
    frame.set_linewidth(0.5)

    #set the attributes of the plot
    params = {'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 6.0,
    'axes.labelsize': 8.0}
    plt.rcParams.update(params)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.3)

    #save the figure
    if save_fp != None:
        plt.savefig(save_fp)
    plt.show()

def get_ARR_plots(xhat_ARRs, bootstrapped_ARRs, cohort_names, figsize, dpi, save_fp):
    """
    Plots the absolute risk reduction vs the reduction in inadaquate
    hemostasis probability score (0-100%). Also fills in the region where
    the 95% CI interval generated by bootstrapping is

    Parameters
    ----------
    xhat_ARRs : list
        Each element corresponds to a subset. Each element comprises the 
        absolute risk reduction for each inadaquate hemostasis probability
        score reduction
    bootstrapped_ARRs : list
        Each element corresponds to a subset. Each element is a 2D numpy.array
        where each row corresponds to the absolute risk reduction of a
        bootstrapped sample
    cohort_names : list
        Each element corresponds to a str that represents the name of the subset
    figsize : tuple
        The size of the figure to generate in inches
    dpi : int
        The dots per inch to put into the figure
    save_fp : str
        The file path to save the figure to. If None, no figure is saved
    """
    #make sure the inputs are correct
    assert len(xhat_ARRs) == len(cohort_names)
    assert len(xhat_ARRs) <= 2
    assert len(bootstrapped_ARRs) == len(cohort_names)
    assert len(bootstrapped_ARRs) <= 2

    #set the parameters
    colors = ["#8c0000", "#006b37"]
    facecolors = ['#c52626', "#006b37"]
    markers = ["-", "--"]
    alphas = [0.15, 0.15]
    fig = plt.figure(figsize = figsize, dpi = dpi)
    axes= fig.add_axes([0.15,0.17,0.8,0.8])

    #reverse the arrays so that the first one is on top
    xhat_ARRs_reversed = xhat_ARRs.copy()
    bootstrapped_ARRs_reversed = bootstrapped_ARRs.copy()
    cohort_names_reversed = cohort_names.copy()
    colors_reversed = colors.copy()
    markers_reversed = markers.copy()
    alphas_reversed = alphas.copy()
    facecolors_reversed = facecolors.copy()

    xhat_ARRs_reversed.reverse()
    bootstrapped_ARRs_reversed.reverse()
    cohort_names_reversed.reverse()
    colors_reversed.reverse()
    markers_reversed.reverse()
    alphas_reversed.reverse()
    facecolors_reversed.reverse()

    #plot the ARR
    plt.xlabel("Percent Reduction (%) in IH Probability Score")
    plt.ylabel("ARR (%)")
    plt.xlim((0,100))

    reduction_percentages = np.arange(101)

    lower_bounds = [np.percentile(x, 2.5, axis = 0) for x in bootstrapped_ARRs_reversed]
    upper_bounds = [np.percentile(x, 97.5, axis = 0) for x in bootstrapped_ARRs_reversed]
    for xhat_ARR, lower_bound, upper_bound, cohort_name, color, facecolor, marker, alpha in zip(xhat_ARRs_reversed, lower_bounds, upper_bounds, cohort_names_reversed, colors_reversed, facecolors_reversed, markers_reversed, alphas_reversed):
        axes.fill_between(
            reduction_percentages,
            lower_bound*100, 
            upper_bound*100, 
            label = cohort_name, 
            facecolor = color,
            alpha = alpha,
            interpolate = True,
            )
        axes.plot(reduction_percentages, xhat_ARR*100, marker, color = color, linewidth = 1.25)
    
    #switch the order of the legends and add grid
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1,0]
    axes.grid(True, linewidth = 0.25, color = 'black', alpha = 0.5)
    legend = axes.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc = 2,
        frameon = True,
        framealpha = 1,
        prop={'size': 6}
    )
    frame = legend.get_frame()
    frame.set_linewidth(0.5)

    #set the attributes
    params = {'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 6.0,
    'axes.labelsize': 8.0}
    plt.rcParams.update(params)
    plt.subplots_adjust(hspace = 0.4)

    #save the figure
    if save_fp != None:
        plt.savefig(save_fp)
    plt.show()