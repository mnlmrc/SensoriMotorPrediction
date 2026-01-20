import itertools

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
import statsmodels.formula.api as smf
import statsmodels.api as sm


# def anova(data, dependent_var, between_subjects_vars=None, within_subjects_vars=None):
#     """
#     Perform a traditional ANOVA with the capability to handle both within-subjects (repeated measures)
#     and between-subjects factors.
#
#     Parameters:
#     data (DataFrame): The dataframe containing the data.
#     dependent_var (str): The name of the dependent variable column.
#     between_subjects_vars (list of str or None): List of the names of the between-subjects variable columns.
#     within_subjects_vars (list of str or None): List of the names of the within-subjects variable columns.
#
#     Returns:
#     DataFrame: ANOVA table result.
#     """
#
#     # Validate input
#     if not dependent_var or dependent_var not in data.columns:
#         raise ValueError("Dependent variable is missing or not found in data.")
#     if between_subjects_vars is None:
#         between_subjects_vars = []
#     if within_subjects_vars is None:
#         within_subjects_vars = []
#
#     # Construct the formula
#     independent_vars = between_subjects_vars + within_subjects_vars
#     formula = f'{dependent_var} ~ ' + ' + '.join([f'C({var})' for var in independent_vars])
#
#     # Fit the model
#     model = ols(formula, data=data).fit()
#
#     # Perform ANOVA
#     anova_results = anova_lm(model)
#
#     return anova_results
# Example usage of the function
# mixed_anova_result = mixed_anova(data, 'Value', ['participant_id', 'channel'], 'timepoint', 'participant_id')
# print(mixed_anova_result.summary())
# The function is commented out to prevent it from running without specific factors being chosen.
# Uncomment and adjust the parameters to fit the specifics of your dataset.
# def rm_anova(df, anova_factors, group_factors=None):
#     """
#     Perform repeated measures ANOVA for specified group and ANOVA factors.
#
#     Parameters:
#     df (pd.DataFrame): The DataFrame containing the data.
#     group_factors (list of str): Columns to group by.
#     anova_factors (list of str): Factors to use in the ANOVA.
#
#     Returns:
#     pd.DataFrame: A DataFrame with the ANOVA results.
#     """
#     # Create an empty DataFrame to store results
#     anova_results = pd.DataFrame(columns=['group', 'factor', 'F-value', 'pval', 'df', 'df_resid'])
#
#     # Iterate over each combination of group factors
#     if group_factors is None:
#         # Perform Repeated Measures ANOVA
#         aovrm = AnovaRM(df, depvar='Value', subject='participant_id', within=anova_factors, aggregate_func=np.mean)
#         res = aovrm.fit()
#
#         # Extract results for each factor and interaction
#         for factor in anova_factors + [':'.join(anova_factors)]:
#             if factor in res.anova_table.index:
#                 F_value = res.anova_table.loc[factor, 'F Value']
#                 p_value = res.anova_table.loc[factor, 'Pr > F']
#                 df1 = res.anova_table.loc[factor, 'Num DF']
#                 df2 = res.anova_table.loc[factor, 'Den DF']
#
#                 # Append results to the DataFrame
#                 row = {'factor': factor, 'F-value': F_value, 'pval': p_value, 'df': df1,
#                        'df_resid': df2}
#                 anova_results.loc[len(anova_results)] = row
#     else:
#         for group_vals, df_group in df.groupby(group_factors):
#             # Perform Repeated Measures ANOVA
#             aovrm = AnovaRM(df_group, depvar='Value', subject='participant_id', within=anova_factors, aggregate_func=np.mean)
#             res = aovrm.fit()
#
#             # Extract results for each factor and interaction
#             for factor in anova_factors + [':'.join(anova_factors)]:
#                 if factor in res.anova_table.index:
#                     F_value = res.anova_table.loc[factor, 'F Value']
#                     p_value = res.anova_table.loc[factor, 'Pr > F']
#                     df1 = res.anova_table.loc[factor, 'Num DF']
#                     df2 = res.anova_table.loc[factor, 'Den DF']
#
#                     # Append results to the DataFrame
#                     row = {'group': group_vals, 'factor': factor, 'F-value': F_value, 'pval': p_value, 'df': df1,
#                            'df_resid': df2}
#                     anova_results.loc[len(anova_results)] = row
#
#     return anova_results


def rm_anova(df, depvar, subject, anova_factors, include_interactions=True):
    """
    Adjusted function to perform repeated measures ANOVA using linear mixed models,
    with appropriate handling for interactions.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - depvar (str): The dependent variable column name.
    - subject (str): The subject identifier column name.
    - anova_factors (list of str): Factors to use in the ANOVA.
    - include_interactions (bool, optional): Whether to include interactions between ANOVA factors.

    Returns:
    - pd.DataFrame: A DataFrame with the ANOVA results.
    """

    # Prepare the fixed effects part of the formula
    fixed_effects_formula = ' + '.join(anova_factors)
    if include_interactions:
        interaction_terms = ' + '.join([':'.join(pair) for pair in itertools.combinations(anova_factors, 2)])
        fixed_effects_formula += ' + ' + interaction_terms

    # Construct the full model formula
    model_formula = f"{depvar} ~ {fixed_effects_formula}"

    # Fit the mixed model
    model = smf.ols(model_formula, df)
    model_fit = model.fit()
    anova_results = sm.stats.anova_lm(model_fit, typ=2)

    return anova_results


def pairwise(df, factor, dep_var='Value', alpha=0.05):
    """
    Perform post hoc tests for a significant factor in repeated measures ANOVA.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    factor (str): The factor for which to perform post hoc tests.
    alpha (float): Significance level for the tests.

    Returns:
    pd.DataFrame: A DataFrame with the post hoc test results.
    """
    posthoc_results = pd.DataFrame(columns=['group1', 'group2', 'stat', 'pval', 'p-adj'])

    # Get unique levels of the factor
    levels = df[factor].unique()

    # Perform pairwise tests
    for (level1, level2) in itertools.combinations(levels, 2):
        group1 = df[df[factor] == level1][dep_var]
        group2 = df[df[factor] == level2][dep_var]

        # Perform the paired t-test
        stat, p = ttest_rel(group1, group2)

        # Bonferroni correction
        p_adj = p * len(levels) / 2  # Adjust for the number of comparisons

        # Check against alpha
        # if p_adj < alpha:
        row = {'group1': level1, 'group2': level2, 'stat': stat, 'pval': p, 'p-adj': p_adj}
        posthoc_results.loc[len(posthoc_results)] = row

    return posthoc_results


class Stat:

    def __init__(self, data, channels, conditions, labels):

        self.data = data  # list of instances of class Dataset3D
        self.channels = channels
        self.conditions = conditions
        self.labels = labels

    # def make_df(self, labels):
    #     """
    #
    #     """
    #     n_timepoints = self.data[0].timepoints
    #     df = pd.DataFrame(columns=['participant_id', 'condition', 'channel', 'timepoint', 'Value'])
    #     for p, p_data in enumerate(self.data):
    #         Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
    #         M, _, cond_names = av_within_participant(p_data.measurements, Z, cond_name=labels)
    #         channels = p_data.channel_descriptors['channels']
    #         for tp in range(n_timepoints):
    #             for ch, channel in enumerate(channels):
    #                 for c, cond in enumerate(cond_names):
    #                     df.loc[len(df)] = {
    #                         'participant_id': p_data.descriptors['participant_id'],
    #                         'condition': cond,
    #                         'channel': channel,
    #                         'timepoint': tp,
    #                         'Value': M[c, ch, tp]
    #                     }
    #     return df

    def make_df(self, labels):
        """

        """
        df = pd.DataFrame(columns=['participant_id', 'condition', 'channel', 'timepoint', 'Value'])
        for p, p_data in enumerate(self.data):
            print(f'processing participant: {p_data.descriptors["participant_id"]}')
            cond = np.unique(p_data.obs_descriptors['cond_vec'])
            n_trials = p_data.measurements.shape[0]
            n_channels = p_data.measurements.shape[1]
            n_timepoints = p_data.measurements.shape[2]
            for tr in range(n_trials):
                for ch in range(n_channels):
                    for tp in range(n_timepoints):
                        df.loc[len(df)] = {
                            'participant_id': p_data.descriptors['participant_id'],
                            'condition': labels[np.where(cond == (p_data.obs_descriptors['cond_vec'][tr]))[0][0]],
                            'channel': p_data.channel_descriptors['channels'][ch],
                            'timepoint': tp,
                            'Value': p_data.measurements[tr, ch, tp]
                        }
        return df

    # def rm_anova(self, data, labels):
    #     n_conditions = len(self.conditions)
    #     n_timepoints = data.shape[-1]
    #
    #     # fval = np.zeros((n_conditions, n_timepoints))
    #     # num_df = np.zeros((n_conditions, n_timepoints))
    #     # den_df = np.zeros((n_conditions, n_timepoints))
    #     # pval = np.zeros((n_conditions, n_timepoints))
    #
    #     for c, cond in enumerate(self.conditions):
    #         for tp in range(n_timepoints):
    #             n_participants = data.shape[0]
    #             n_labels = len(labels[c])
    #             subject_ids = np.repeat(np.arange(n_participants), n_labels)
    #             data_df = data[:, c, :, tp].flatten()
    #             labels_df = np.array([labels[c]] * n_participants).flatten()
    #
    #             df = pd.DataFrame({'Subject': subject_ids, 'Condition': labels_df, 'Score': data_df})
    #
    #             aovrm = AnovaRM(df, 'Score', 'Subject', within=['Condition'])
    #             res = aovrm.fit()
    #
    #             fval[c, tp] = res.anova_table['F Value'].to_numpy()[0]
    #             num_df[c, tp] = res.anova_table['Num DF'].to_numpy()[0]
    #             den_df[c, tp] = res.anova_table['Den DF'].to_numpy()[0]
    #             pval[c, tp] = res.anova_table['Pr > F'].to_numpy()[0]
    #
    #     return fval, num_df, den_df, pval
    # print(res.summary())

    # def _make_df_for_anova(self, data, condition):

    # def av_across_participants(self):
    #
    #     n_conditions = len(self.conditions)
    #
    #     channels_dict = {ch: [] for ch in self.channels}
    #     N = len(self.data)
    #     for p_data in self.data:
    #         Z = indicator(p_data.obs_descriptors['cond_vec']).astype(bool)
    #         M = av_within_participant(p_data.measurements, Z)
    #
    #         for ch in self.channels:
    #             if ch in p_data.channel_descriptors['channels']:
    #                 channel_index = p_data.channel_descriptors['channels'].index(ch)
    #                 channels_dict[ch].append(M[:, channel_index])
    #
    #     Mean, SD, SE = {}, {}, {}
    #     for ch in self.channels:
    #         channel_data = np.array(channels_dict[ch])
    #         channels_dict[ch] = channel_data.reshape((channel_data.shape[0], n_conditions, int(channel_data.shape[1] / n_conditions), channel_data.shape[2]))
    #         Mean[ch] = np.mean(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
    #                                                           channel_data.shape[2]))
    #         SD[ch] = np.std(channel_data, axis=0).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
    #                                                        channel_data.shape[2]))
    #         SE[ch] = (SD[ch] / np.sqrt(N)).reshape((n_conditions, int(channel_data.shape[1] / n_conditions),
    #                                                 channel_data.shape[2]))
    #
    #     return Mean, SD, SE, channels_dict
