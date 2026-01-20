from scipy.stats import ttest_1samp
import pandas as pd

def ttest_1samp_df(df, group_col, value_col, popmean=0, alternative='two-sided'):
    tvals, pvals = [], []
    for name, group in df.groupby(group_col)[value_col]:
        stat = ttest_1samp(group, popmean, nan_policy='omit', alternative=alternative)
        tvals.append({'group': name, 't': stat.statistic, 'p': stat.pvalue})
    return pd.DataFrame(tvals).set_index('group')
