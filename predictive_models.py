from sklearn.linear_model import LogisticRegression  
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import random
import matplotlib.pyplot as plt
import smote_variants
import json


class Predictive_Model(LogisticRegression):
    """
    Class to build predictive models for inadequate hemostatic efficacy and
    outcome 

    Attributes
    ----------
    feature_names : list
        List of column names to extract for features to use in the model
    outcome_name : str 
        Name of the column to extract as the outcome to use in the model
    train_data : pandas.DataFrame
        The data that will be used to train the model
    smote : Boolean
        Whether to perform a minority class oversampler on the data before
        training
    smote_proporition : float
        The proportion the minority class size should be to that of the 
        majority class. Use 1.0 for equal sizes between the minority and 
        majority class
    random_state : float
        The seed used when fixing the Logistic Regression model training and
        smote

    Methods
    -------
    predict_proba(self, data)
        Get the prediction probability of each class
    get_CV_aurocc(self, seed = 0, return_mean_auc_not_auc_of_mean_tpr_fpr = True, nsplits = 10, roundTo = 3)
        Get the cross validated area under the receiver operating characteristic
        curve (CV-AUROC) of the IHE model.
    get_coefficients(self)
        Gets a Dataframe with the coefficient names, log odds coefficients, and
        odd ratios used to create the IHE model
    """
    def __init__(self, feature_names, outcome_name, train_data, smote = False, smote_proportion = 1.0, random_state = 0):
        """
        Parameters
        ----------
        feature_names : list
            List of column names to extract for features to use in the model
        outcome_name : str 
            Name of the column to extract as the outcome to use in the model
        train_data : pandas.DataFrame
            The data that will be used to train the model
        smote : Boolean
            Whether to perform a minority class oversampler on the data before
            training
        smote_proporition : float
            The proportion the minority class size should be to that of the 
            majority class. Use 1.0 for equal sizes between the minority and 
            majority class
        random_state : float
            The seed used when fixing the Logistic Regression model training and
            smote
        """
        super().__init__(random_state = random_state)
        self.feature_names = feature_names
        self.feature_full_name = []
        self.outcome_name = outcome_name
        self.train_data = train_data
        self.smote = smote
        self.smote_proportion = smote_proportion
        self.smote_function = smote_variants.ProWSyn
        self.X, self.y = train_data[feature_names], train_data[outcome_name]
        
        if smote == False:
            self.fitX, self.fity = self.X, self.y
        else:
            self.fitX, self.fity  = self.smote_function(proportion = smote_proportion, random_state = random_state).fit_resample(
                np.array(train_data[feature_names]),
                np.array(train_data[outcome_name]))
            
        self.fit(self.fitX, self.fity)
    
    def predict_proba(self, data):
        """
        Get the prediction probability of each class

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset on which the model should predict probabilites on. The
            dataset may contain more features than used to train the model as 
            only the features used to train the model are selected

        Returns
        -------
        numpy.ndarray
            2D array of probabilities. Each row represents a row from 'data' or
            patient. Use the second column for the probabilities of the positive
            class or probability for IHE
        """

        return super().predict_proba(data[self.feature_names])
    
    def get_CV_aurocc(self, seed = 0, return_mean_auc_not_auc_of_mean_tpr_fpr = True, nsplits = 10, roundTo = 3):
        """
        Get the cross validated area under the receiver operating characteristic
        curve (CV-AUROC) of the IHE model. This splits the dataset saved when
        initializing the model into k partions. k-1 splits are used as training,
        while the unused split is used for the testing cohort. If a minority 
        oversampler was used, then this is applied on the training cohort before
        training. This is repeated k-1 times and the mean AUROC is returned

        Parameters
        ----------
        seed : int
            Random seed used to set the random k splits and the smote function
        return_mean_auc_not_auc_of_mean_tpr_fpr : Boolean
            If true, the mean of the auroc of the model from each iteration is
            returned. If false, the true positive rate (TPR) and false positive
            rates (FPR) are averaged across all k models. The area under the 
            curve of the mean TPR and FPR is then calculated. Set this to false
            if you want to generate a CV-AUROC plot and have the AUROC reflect
            the area under the curve of the mean curve
        nsplits : int
            The number of k splits to perform
        roundTo : int
            The number of decimal points to round the returned CV-AUROC 
        
        Returns
        -------
        float
            The CV-AUROC of the IHE model
        list
            The mean FPR across k splits
        list
            The mean TPR across k splits
        """
        X,y = np.array(self.X), np.array(self.y)
        random.seed(seed)
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        #get the splits
        cv = StratifiedKFold(n_splits=nsplits, random_state= random.randint(0,1000), shuffle=True)
        #get fpr, tpr, auroc of each k-split
        for train_idx, test_idx, in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            if self.smote:
                X_train, y_train = self.smote_function(proportion = self.smote_proportion, random_state = random.randint(0,1000)).fit_resample(X_train, y_train)

            clf = LogisticRegression().fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]

            #get the auroc of the split
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            #interpolate the tpr so that there is a full range from 1-100
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0

        if return_mean_auc_not_auc_of_mean_tpr_fpr:
            cv_auroc = np.mean(aucs)
        else:
            cv_auroc = auc(mean_fpr, mean_tpr)

        return round(cv_auroc, roundTo), mean_fpr, mean_tpr

    def get_coefficients(self):
        coeff_df = pd.DataFrame()
        
        coeff_df["Feature Name"] = self.feature_full_name[:-1] + ['Cons']
        coeff_df['Column Name'] = self.feature_names + ['Cons']
        
        coef = self.coef_.flatten().tolist()
        coef.append(self.intercept_[0])
        coeff_df['Coefficients'] = coef
        
        coeff_df['Odds Ratio'] = np.exp(coef)
        
        return coeff_df

class IHE_Model(Predictive_Model):
    """
    Class to build a model to predict for inadequate hemostatic efficacy (IHE)

    Attributes
    ----------
    train_data : pandas.DataFrame
        The data that will be used to train the model
    smote : Boolean
        Whether to perform a minority class oversampler on the data before
        training
    smote_proporition : float
        The proportion the minority class size should be to that of the 
        majority class. Use 1.0 for equal sizes between the minority and 
        majority class
    random_state : float
        The seed used when fixing the Logistic Regression model training and
        smote
    """
    def __init__(self, train_data, smote = True, smote_proportion = 1.0, random_state = 0):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            The data that will be used to train the model
        smote : Boolean
            Whether to perform a minority class oversampler on the data before
            training
        smote_proporition : float
            The proportion the minority class size should be to that of the 
            majority class. Use 1.0 for equal sizes between the minority and 
            majority class
        random_state : float
            The seed used when fixing the Logistic Regression model training and
            smote
        """
        feature_dictionary = json.load(open("./variable_names.json", mode="r"))
        ihe_feature_names, ihe_full_names = _pull_ihe_features_from_json(feature_dictionary)
        super().__init__(ihe_feature_names[:-1], ihe_feature_names[-1], train_data, smote = smote, smote_proportion = smote_proportion, random_state = random_state)
        self.feature_full_name = ihe_full_names

class Poor_Outcome_Model(Predictive_Model):
    """
    Class to build a model to predict for poor outcome (mRS 4-6)

    Attributes
    ----------
    train_data : pandas.DataFrame
        The data that will be used to train the model
    random_state : float
        The seed used when fixing the Logistic Regression model training and
        smote
    """
    def __init__(self, train_data, random_state=0):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            The data that will be used to train the model
        random_state : float
            The seed used when fixing the Logistic Regression model training and
            smote
        """
        feature_dictionary = json.load(open("./variable_names.json", mode="r"))
        feature_names, feature_full_name = _pull_outcome_features_from_json(feature_dictionary)
        super().__init__(feature_names[:-1], feature_names[-1], train_data, smote = False, smote_proportion = None, random_state = random_state)
        self.feature_full_name = feature_full_name
        self.ihe_probability_feature = feature_dictionary["Probability for IHE"]

def _pull_ihe_features_from_json(feature_dictionary):
    """
    Gets the feature names of the IHE model that are saved in the CSV file and
    what those feature names represents. The last element represents the outcome
    name

    Parameters
    ----------
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    
    Returns
    -------
    list
        Feature names that are saved in the csv file
    list
        Full name of the respective features
    """
    feature_names_csv = [
        feature_dictionary["Log-transformed initial ICH volume"],
        feature_dictionary["Log-transformed hours from LKW to hospital arrival"],
        feature_dictionary["Warfarin Use"],
        feature_dictionary["FXa inhibitor use"],
        feature_dictionary["CTA spot sign"],
        feature_dictionary["Other Coagulopathy"],
        feature_dictionary["Single Antiplatelet"],
        feature_dictionary["Double Antiplatelet"],
        feature_dictionary["Systolic BP on Arrival"],
        feature_dictionary["IHE"],
    ]

    feature_names_full = [
        "Log-transformed initial ICH volume",
        "Log-transformed hours from LKW to hospital arrival",
        "Warfarin Use",
        "FXa inhibitor use",
        "CTA spot sign",
        "Other Coagulopathy",
        "Single Antiplatelet",
        "Double Antiplatelet",
        "Systolic BP on Arrival",
        "IHE",
    ]
    return feature_names_csv, feature_names_full

def _pull_outcome_features_from_json(feature_dictionary):
    """
    Gets the feature names of the outcome model that are saved in the CSV file
    and what those feature names represents. The last element represents the outcome
    name

    Parameters
    ----------
    feature_dictionary : dict
        Dictionary where keys are the full names of the feature and the values
        are the feature names saved in the csv file
    
    Returns
    -------
    list
        Feature names that are saved in the csv file
    list
        Full name of the respective features
    str
        Name of the IHE probability name to use
    """

    feature_names_full = [
        "Log-transformed Age",
        "Log-transformed initial ICH volume",
        "Initial GCS Score: 5-12",
        "Initial GCS Score: 3-4",
        "IVH",
        "Probability for IHE",
        "Poor Outcome at 3 months"
    ]

    feature_names_csv = [feature_dictionary[k] for k in feature_names_full]

    return feature_names_csv, feature_names_full,