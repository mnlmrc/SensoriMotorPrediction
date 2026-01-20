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
