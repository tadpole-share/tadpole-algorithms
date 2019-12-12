# %%
import os
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn.svm as svm
from sklearn.utils import resample
import scipy as sp
import scipy.interpolate as interpolate
import sys

sys.path.append('..')
from evaluation import evalOneSubmissionExtended as eos



def preprocess(df):
    # %%
    # Remove some features
    d3 = True
    if d3:
        df.drop(['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT'], axis=1)
    else:
        df.drop(['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'APOE4'], axis=1)
    df['Ventricles_ICV'] = df['Ventricles'].values / df['ICV_bl'].values
    df.replace({'DXCHANGE': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})
    df = df.rename(columns={"DXCHANGE": "Diagnosis"})

    # Force values to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop all columns that are 100% NaN
    df = df.drop(columns=df.columns[df.isna().all()])
    return df

# %%
def bootstrap(Dtrainmat_ADAS13, Y_FutureADAS13_norm, Dtestmat, n_bootstraps=100, confidence=0.50):
    input_matrix = np.append(Dtrainmat_ADAS13, Y_FutureADAS13_norm[:, None], axis=1)
    y_ADAS13_norm = np.zeros((n_bootstraps, Dtestmat.shape[0]))
    for i in range(0, n_bootstraps):
        print(f"Bootstrap :{str(i)} ")
        input_matrix_resampled = resample(input_matrix, random_state=i)

        Dtrainmat_ADAS13 = input_matrix_resampled[:, :-1]
        Y_FutureADAS13_norm = input_matrix_resampled[:, -1]

        reg = svm.SVR(kernel='rbf')
        reg.fit(Dtrainmat_ADAS13, Y_FutureADAS13_norm)
        y_ADAS13_norm[i, :] = reg.predict(Dtestmat)

    n = len(y_ADAS13_norm)
    se = sp.stats.sem(y_ADAS13_norm)  # Standard error
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)  # CI

    # m-h and m+h give confidence interval
    return h


# %%
# Settings
leaderboard = 0
d3 = 1

# %%
# Define input directory
str_exp = os.path.dirname(
    os.path.realpath('__file__'))  # here we added __file__ in quotes to resolve name not found error
os.chdir(str_exp)
# %%
# Define output IntermediateData
str_out_final = os.path.join(str_exp, 'IntermediateData', 'TADPOLE_Submission_EMC-EB1.csv')
# %%
# Define inputs
# train_data
str_in = os.path.join(str_exp, 'IntermediateData', 'LongTADPOLE.csv')
# test_data
predict_file = os.path.join(str_exp, 'IntermediateData', 'ToPredict_D2.csv')
# ground truth
ref_file = os.path.join(str_exp, 'IntermediateData', 'D4_dummy.csv')

# %%
if leaderboard:
    str_out_final = str_out_final.replace('_Submission', '_Submission_Leaderboard')
    predict_file = os.path.join(str_exp, 'IntermediateData', 'ToPredict.csv')
    ref_file = os.path.join(str_exp, 'Data', 'TADPOLE_LB4.csv')
# %%
if d3:
    str_in = str_in.replace('LongTADPOLE', 'LongTADPOLE_D1')
    str_out_final = str_out_final.replace('EMC', 'EMC-D3')
    predict_file = os.path.join(str_exp, 'IntermediateData', 'LongTADPOLE_D3.csv')
# %%
if leaderboard and d3:
    print('Does not work yet!!!!!')
    predict_file = os.path.join(str_exp, 'IntermediateData', 'LongTADPOLE_LB3.csv')

# %%
# Read data
D = pd.read_csv(Path('/home/tom/Projects/tadpole/jupyter/data/TADPOLE_D1_D2.csv'))
D = D[:10]
D = preprocess(D)
# %%
# Get Future Measurements
Y_FutureADAS13_temp = D['ADAS13'].copy()
Y_FutureADAS13_temp[:] = np.nan
Y_FutureVentricles_ICV_temp = D['Ventricles_ICV'].copy()
Y_FutureVentricles_ICV_temp[:] = np.nan
Y_FutureDiagnosis_temp = D['Diagnosis'].copy()
Y_FutureDiagnosis_temp[:] = np.nan
RID = D['RID'].copy()
uRIDs = np.unique(RID)
for i in range(len(uRIDs)):
    idx = RID == uRIDs[i]
    idx_copy = np.copy(idx)
    idx_copy[np.where(idx)[-1][-1]] = False
    Y_FutureADAS13_temp[idx_copy] = D.loc[idx, 'ADAS13'].values[1:]
    Y_FutureVentricles_ICV_temp[idx_copy] = D.loc[idx, 'Ventricles_ICV'].values[1:]
    Y_FutureDiagnosis_temp[idx_copy] = D.loc[idx, 'Diagnosis'].values[1:]
D = D.drop(['RID'], axis=1)

# %%
# Get Features for selection
if d3:
    percentage = .50
    idx_fewmissing = pd.isnull(D).select_dtypes(include=['bool']).sum(axis=0) < percentage * D.shape[0]
    Dtrain = D.loc[:, idx_fewmissing].copy()
else:
    # idx_selected = pd.read_csv(os.path.join(str_exp, 'IntermediateData', 'FeatureIndices.csv'))
    idx_selected = pd.read_csv(os.path.join(str_exp, 'IntermediateData', 'FeatureIndices.csv'))
    idx_Feats = idx_selected['x'].values
    i1 = 200
    idx_Feats_sel = idx_Feats[:i1] + 1  # +1 as I don't remove Diagnosis column
    idx_Feats_sel = np.append(0, idx_Feats_sel)  # Include diagnosis
    idx_Feats_sel = np.append(idx_Feats_sel, D.shape[1] - 1)  # Include ventricles_icv
    Dtrain = D.iloc[:, idx_Feats_sel].copy()

# %%
# Fill nans in feature matrix by older values
urid = np.unique(RID)
for j1 in range(len(urid)):
    idx_rid = RID == urid[j1]
    dtr = Dtrain[idx_rid].values
    for k1 in range(dtr.shape[1]):
        if np.sum(np.isnan(dtr[:, k1])) < len(dtr[:, k1]):
            idx_nan = np.isnan(dtr[:, k1])
            indices_val = np.where(np.logical_not(idx_nan))[0]
            fk1 = interpolate.interp1d(indices_val, dtr[indices_val, k1], kind='zero', bounds_error=False,
                                       fill_value=(dtr[indices_val, k1][0], dtr[indices_val, k1][-1]))
            dtr[:, k1] = fk1(range(len(dtr[:, k1])))
    Dtrain.loc[idx_rid, :] = dtr


# Fill other nans
Dtrain = Dtrain.fillna(Dtrain.mean())


# %%
# Fill nans in feature matrix
Dtrainmat = Dtrain.values  # Method .as_matrix will be removed in a future version. Use .values instead.



h = list(Dtrain)
m = []
s = []
for i in range(Dtrainmat.shape[1]):
    m.append(np.nanmean(Dtrainmat[:, i]))
    s.append(np.nanstd(Dtrainmat[:, i]))
    Dtrainmat[np.isnan(Dtrainmat[:, i]), i] = m[i]
    Dtrainmat[:, i] = (Dtrainmat[:, i] - m[i]) / s[i]

# %%
# Remove NaNs in Diagnosis
idx_last_Diagnosis = np.isnan(Y_FutureDiagnosis_temp)
RID_Diagnosis = RID.copy()
Dtrainmat_Diagnosis = Dtrainmat.copy()
Dtrainmat_Diagnosis = Dtrainmat_Diagnosis[np.logical_not(idx_last_Diagnosis), :]
Y_FutureDiagnosis = Y_FutureDiagnosis_temp[np.logical_not(idx_last_Diagnosis)].copy()
RID_Diagnosis = RID_Diagnosis[np.logical_not(idx_last_Diagnosis)]

idx_last_ADAS13 = np.isnan(Y_FutureADAS13_temp)
RID_ADAS13 = RID.copy()
Dtrainmat_ADAS13 = Dtrainmat.copy()
Dtrainmat_ADAS13 = Dtrainmat_ADAS13[np.logical_not(idx_last_ADAS13), :]
RID_ADAS13 = RID_ADAS13[np.logical_not(idx_last_ADAS13)]

Y_FutureADAS13 = Y_FutureADAS13_temp[np.logical_not(idx_last_ADAS13)].copy()
m_FutureADAS13 = np.nanmean(Y_FutureADAS13)
s_FutureADAS13 = np.nanstd(Y_FutureADAS13)
Y_FutureADAS13_norm = (Y_FutureADAS13 - m_FutureADAS13) / s_FutureADAS13

idx_last_Ventricles_ICV = np.isnan(Y_FutureVentricles_ICV_temp)
RID_Ventricles_ICV = RID.copy()
Dtrainmat_Ventricles_ICV = Dtrainmat.copy()
Dtrainmat_Ventricles_ICV = Dtrainmat_Ventricles_ICV[np.logical_not(idx_last_Ventricles_ICV), :]
RID_Ventricles_ICV = RID_Ventricles_ICV[np.logical_not(idx_last_Ventricles_ICV)]

Y_FutureVentricles_ICV = Y_FutureVentricles_ICV_temp[np.logical_not(idx_last_Ventricles_ICV)].copy()
m_FutureVentricles_ICV = np.nanmean(Y_FutureVentricles_ICV)
s_FutureVentricles_ICV = np.nanstd(Y_FutureVentricles_ICV)
Y_FutureVentricles_ICV_norm = (Y_FutureVentricles_ICV - m_FutureVentricles_ICV) / s_FutureVentricles_ICV
# %%
print('Training methods')
# Train SVM for diagnosis
import sklearn.svm as svm

clf = svm.SVC(kernel='rbf', C=0.5, class_weight='balanced', probability=True)
clf.fit(Dtrainmat_Diagnosis, Y_FutureDiagnosis)

# Train SVR for ADAS
reg_ADAS13 = svm.SVR(kernel='rbf', C=0.5)
reg_ADAS13.fit(Dtrainmat_ADAS13, Y_FutureADAS13_norm)

# Train SVR for Ventricles
reg_Ventricles_ICV = svm.SVR(kernel='rbf', C=0.5)
reg_Ventricles_ICV.fit(Dtrainmat_Ventricles_ICV, Y_FutureVentricles_ICV_norm)
# %%
# Create TestSet
if d3:
    D3 = pd.read_csv(predict_file)

    # Remove some features
    D31 = D3[['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT']].copy()
    D3 = D3.drop(['EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT'], axis=1)
    D3['Ventricles_ICV'] = D3['Ventricles'].values / D3['ICV_bl'].values
    S = D3['RID'].copy()
    D3 = D3.drop(['RID'], axis=1)

    D3train = D3.loc[:, idx_fewmissing].copy()

    # Fill nans in feature matrix
    Dtestmat = D3train.values
    h = list(D3)
    for i in range(Dtestmat.shape[1]):
        Dtestmat[np.isnan(Dtestmat[:, i]), i] = m[i]
        Dtestmat[:, i] = (Dtestmat[:, i] - m[i]) / s[i]

else:
    S = pd.read_csv(predict_file, header=None)
    S = S.values

    Dtestmat = np.zeros((len(S), Dtrainmat.shape[1]))
    for i in range(len(S)):
        idx_S = RID.values == S[i]
        Dtestmat[i, :] = Dtrainmat[np.where(idx_S)[0][-1], :]

# %%
print('Testing methods')
# Test SVM for Diagnosis
p = clf.predict_proba(Dtestmat)  # _Diagnosis

# Test SVR for ADAS
y_ADAS13_norm = reg_ADAS13.predict(Dtestmat)
h = bootstrap(Dtrainmat_ADAS13, Y_FutureADAS13_norm, Dtestmat)

y_ADAS13_norm[y_ADAS13_norm * s_FutureADAS13 + m_FutureADAS13 < 0] = 0
y_ADAS13 = y_ADAS13_norm * s_FutureADAS13 + m_FutureADAS13
y_ADAS13_lower = (y_ADAS13_norm - h) * s_FutureADAS13 + m_FutureADAS13
y_ADAS13_lower[y_ADAS13_lower < 0] = 0
y_ADAS13_upper = (y_ADAS13_norm + h) * s_FutureADAS13 + m_FutureADAS13
y_ADAS13_upper[y_ADAS13_upper < 0] = 0

# %%
# Test SVR for Ventricles
y_Ventricles_ICV_norm = reg_Ventricles_ICV.predict(Dtestmat)
h = bootstrap(Dtrainmat_Ventricles_ICV, Y_FutureVentricles_ICV_norm, Dtestmat)

y_Ventricles_ICV_norm[y_Ventricles_ICV_norm * s_FutureVentricles_ICV + m_FutureVentricles_ICV < 0] = 0
y_Ventricles_ICV = y_Ventricles_ICV_norm * s_FutureVentricles_ICV + m_FutureVentricles_ICV
y_Ventricles_ICV_lower = (y_Ventricles_ICV_norm - h) * s_FutureVentricles_ICV + m_FutureVentricles_ICV
y_Ventricles_ICV_lower[y_Ventricles_ICV_lower < 0] = 0
y_Ventricles_ICV_upper = (y_Ventricles_ICV_norm + h) * s_FutureVentricles_ICV + m_FutureVentricles_ICV
y_Ventricles_ICV_upper[y_Ventricles_ICV_upper < 0] = 0

o = np.column_stack((S, S, S, p, y_ADAS13, y_ADAS13_lower, y_ADAS13_upper, y_Ventricles_ICV, y_Ventricles_ICV_lower,
                     y_Ventricles_ICV_upper))
count = 0
if leaderboard:
    years = [str(a) for a in range(2010, 2018)]
else:
    years = [str(a) for a in range(2018, 2023)]
months = [str(a).zfill(2) for a in range(1, 13)]
ym = [y + '-' + mo for y in years for mo in months]  # TADPOLE_D1_D2
if leaderboard:
    ym = ym[4:-8]
nr_pred = len(ym)
o1 = np.zeros((o.shape[0] * nr_pred, o.shape[1]))
ym1 = [a for b in range(0, len(S)) for a in ym]
for i in range(len(o)):
    o1[count:count + nr_pred] = o[i]
    o1[count:count + nr_pred, 1] = range(1, nr_pred + 1)
    count = count + nr_pred

# %%
# Save output
output = pd.DataFrame(o1, columns=['RID', 'Forecast Month', 'Forecast Date', 'CN relative probability',
                                   'MCI relative probability', 'AD relative probability', 'ADAS13',
                                   'ADAS13 50% CI lower', 'ADAS13 50% CI upper', 'Ventricles_ICV',
                                   'Ventricles_ICV 50% CI lower', 'Ventricles_ICV 50% CI upper'])
output['Forecast Month'] = output['Forecast Month'].astype(int)
output['Forecast Date'] = ym1

output.to_csv(str_out_final, header=True, index=False)

# Evaluate output
R = pd.read_csv(ref_file)

mAUC, bca, adasMAE, ventsMAE, adasWES, ventsWES, adasCPA, ventsCPA, adasEstim, trueADASFilt = eos.evalOneSub(R, output)

print('Diagnosis:')
print('mAUC = ' + "%0.3f" % mAUC, )
print('BAC = ' + "%0.3f" % bca)
print('ADAS:')
print('MAE = ' + "%0.3f" % adasMAE, )
print('WES = ' + "%0.3f" % adasWES, )
print('CPA = ' + "%0.3f" % adasCPA)
print('VENTS:')
print('MAE = ' + "%0.3e" % ventsMAE, )
print('WES = ' + "%0.3e" % ventsWES, )
print('CPA = ' + "%0.3f" % ventsCPA)

# %%
