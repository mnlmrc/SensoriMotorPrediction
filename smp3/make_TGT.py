import pandas as pd
import numpy as np

experiment = 'smp3'
sn = 100
nruns = 10

for run in range(nruns):

    # Load the file
    df = pd.read_csv("target/smp3_template.tgt", sep="\t")

    # Shuffle planTime independently
    df['planTime'] = np.random.permutation(df['planTime'].values)

    # Shuffle iti, TrigBaseline, stimTrigBaseline together
    shuffled_group1 = df[['iti', 'TrigBaseline', 'stimTrigBaseline']].sample(frac=1).reset_index(drop=True)
    df[['iti', 'TrigBaseline', 'stimTrigBaseline']] = shuffled_group1

    # Shuffle the rest of the fields together
    shuffled_group2 = df[['subNum', 'cueID', 'stimFinger', 'execMaxTime', 'feedbackTime', 'trialLabel', 'TrigExec',
                          'TrigPlan', 'stimTrigExec', 'stimTrigPlan']].sample(frac=1).reset_index(drop=True)
    df[['subNum', 'cueID', 'stimFinger', 'execMaxTime', 'feedbackTime', 'trialLabel', 'TrigExec',
                          'TrigPlan', 'stimTrigExec', 'stimTrigPlan']] = shuffled_group2

    # Compute startTime
    start_times = [10000]
    for i in range(1, len(df)):
        prev = df.loc[i-1]
        prev_sum = prev['planTime'] + prev['execMaxTime'] + prev['feedbackTime'] + prev['iti']
        start_times.append(start_times[-1] + prev_sum + 500)
    df['startTime'] = start_times

    # Save result
    df.to_csv(f"target/{experiment}_{sn}_{run+1:02}.tgt", sep="\t", index=False)
