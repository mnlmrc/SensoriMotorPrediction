import pandas as pd
import numpy as np

experiment = 'smp3'
sn = 102
nruns = 10

df_no_tms = pd.read_csv("/cifs/diedrichsen/data/SensoriMotorPrediction/smp3/target/smp3_template_no_tms.tgt", sep="\t")
df_no_tms_shuffled = df_no_tms.sample(frac=1).reset_index(drop=True)

batch_size = 4

b = 0

for run in range(nruns):
    # Load the file
    df = pd.read_csv("/cifs/diedrichsen/data/SensoriMotorPrediction/smp3/target/smp3_template.tgt", sep="\t")

    # Shuffle planTime independently
    df['planTime'] = np.random.permutation(df['planTime'].values)

    # Shuffle iti, TrigBaseline, stimTrigBaseline together
    shuffled_group1 = df[['iti', 'TrigBaseline', 'stimTrigBaseline']].sample(frac=1).reset_index(drop=True)
    df[['iti', 'TrigBaseline', 'stimTrigBaseline']] = shuffled_group1

    # Shuffle the rest of the fields together
    shuffled_group2 = df[['subNum', 'cueID', 'stimFinger', 'execMaxTime', 'feedbackTime', 'trialLabel', 'TrigExec',
                          'TrigPlan', 'stimTrigExec']].sample(frac=1).reset_index(drop=True)
    df[['subNum', 'cueID', 'stimFinger', 'execMaxTime', 'feedbackTime', 'trialLabel', 'TrigExec',
                          'TrigPlan', 'stimTrigExec']] = shuffled_group2
    df['stimTrigPlan'] = df['planTime'] - 100

    df_no_tms_batch = df_no_tms_shuffled[b:b + batch_size]
    b = + batch_size

    df = pd.concat([df, df_no_tms_batch])
    df = df.sample(frac=1).reset_index(drop=True)

    # Compute startTime
    start_times = [10000]
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        prev_sum = prev['planTime'] + prev['execMaxTime'] + prev['feedbackTime'] + prev['iti']
        start_times.append(start_times[-1] + prev_sum + 500)
    df['startTime'] = start_times

    # Save result
    df.to_csv(f"/cifs/diedrichsen/data/SensoriMotorPrediction/smp3/target/{experiment}_{sn}_{run+1:02}.tgt",
              sep="\t", index=False)
