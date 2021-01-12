#df = pd.read_csv("Outputs/emceb/forecast_df_d2.csv")
##df = df.sort_values(by=['RID'])
#total = df[["CN relative probability", "MCI relative probability", "AD relative probability"]]

#df = pd.read_csv("Outputs/emc1/forecast_df_d2.csv")
#total = total + df[["CN relative probability", "MCI relative probability", "AD relative probability"]]

#total = total/2

#df[["CN relative probability", "MCI relative probability", "AD relative probability"]] = total[["CN relative probability", "MCI relative probability", "AD relative probability"]]

##os.chdir('..')
#data_path_eval = ("data/TADPOLE_D4_corr.csv")
#data_df_eval = pd.read_csv(data_path_eval)
#eval_df = data_df_eval

#from tadpole_algorithms.evaluation import print_metrics
#from tadpole_algorithms.evaluation import evaluate_forecast

#dictionary=evaluate_forecast(eval_df, df)
#print_metrics(dictionary)

#print(df.size())
###################################################################################


import pandas as pd
import os
import numpy as np

### Load dataframes
emc_eb = pd.read_csv("Outputs/emceb/forecast_df_d2.csv")
emc1 = pd.read_csv("Outputs/emc1/forecast_df_d2.csv")
benchmark_svm = pd.read_csv("Outputs/benchmark_svm/forecast_df_d2.csv")
benchmark_last_visit = pd.read_csv("Outputs/benchmark_last_visit/forecast_df_d2.csv")
df_mexico = pd.read_csv("Outputs/BORREGOSTECMTY/forecast_df_d2.csv")
##df_cbig_rnn = pd.read_csv("Outputs/cbig_rnn/forecast_df_d2.csv")
##df_smnsr = pd.read_csv("Outputs/SMNSR/forecast_df_d2.csv")

dfs = [emc_eb, emc1, benchmark_svm, benchmark_last_visit]
dfs_names = ["df_emceb", "df_emc1", "df_benchmark_svm", "df_benchmark_last_visit"]

### Preprocessing function
def process(df):
    ## Sort dfs
    #df['Forecat Date'] = pd.to_datetime(df['Forecast Date'])
    #df.sort_values(['Forecast Date'], inplace=True, kind='mergesort' )
    
    ## Get diangnosis probabilities
    #emc_eb_old.iloc[:, [4,5,6]]

    ## Get diangnosis probabilities
    df = df[["CN relative probability", "MCI relative probability", "AD relative probability"]]
    return df

def name(name):
    return dfs_names.index(name)

for i, df in enumerate(dfs):
    dfs[i] = process(df)

print("#############")
print(dfs[0].columns)

#df_emceb = process(df_emceb)
#df_emc1 = process(df_emc1)
#df_benchmark_svm = process(df_benchmark_svm)
#df_benchmark_last_visit = process(df_benchmark_last_visit)

#Combining dataframes with MEAN
num = 2
df_mean1 = (dfs[name('df_emceb')] + dfs[name('df_emc1')] + dfs[name('df_benchmark_svm')] + dfs[name('df_benchmark_last_visit')]) / 4
df_mean2 = (dfs[name('df_emceb')] + dfs[name('df_emc1')]) / 2
df_mean3 = (dfs[name('df_emc1')] + dfs[name('df_benchmark_svm')]) / 2
df_mean4 = (dfs[name('df_benchmark_last_visit')] + dfs[name('df_benchmark_svm')]) / 2
df_mean5 = (dfs[name('df_benchmark_last_visit')] + dfs[name('df_benchmark_svm')] + dfs[name('df_emceb')]) / 3

## add other columns
benchmark_last_visit[["CN relative probability", "MCI relative probability", "AD relative probability"]] = df_mean4[["CN relative probability", "MCI relative probability", "AD relative probability"]]
mean4 = benchmark_svm

#os.chdir('..')
data_path_eval = ("data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)
eval_df = data_df_eval

from tadpole_algorithms.evaluation import print_metrics
from tadpole_algorithms.evaluation import evaluate_forecast

dictionary=evaluate_forecast(eval_df, df_mexico)
print_metrics(dictionary)



#emc_eb.iloc[:,[1,3]].isin(emc1.iloc[:,[1,2]])
#currSubjData = forecastDf[currSubjMask]


nrSubj = eval_df.shape[0]
benchmark_svm['Forecast Date'] = pd.to_datetime(benchmark_svm['Forecast Date'])
eval_df['CognitiveAssessmentDate'] = pd.to_datetime(eval_df['CognitiveAssessmentDate'])
for s in range(nrSubj):
    currSubjMask = eval_df['RID'].iloc[s] == benchmark_svm['RID']
    currSubjData = benchmark_svm[currSubjMask]

    timeDiffsScanCog = [eval_df['CognitiveAssessmentDate'].iloc[s] - d for d in currSubjData['Forecast Date']]
    # print('Forecast Date 2',currSubjData['Forecast Date'])
    indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]
    #print(indexMin)


    currSubjData = currSubjData.iloc[[indexMin]]
    print(currSubjData)
    pCN = currSubjData['CN relative probability'].iloc[indexMin]
    pMCI = currSubjData['MCI relative probability'].iloc[indexMin]
    pAD = currSubjData['AD relative probability'].iloc[indexMin]

#print(benchmark_svm)
#print(pCN)
