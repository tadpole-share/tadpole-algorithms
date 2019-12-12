## Classify
import pandas as pd
import numpy as np
import os 
str_exp=os.path.dirname(os.path.realpath(__file__))

os.chdir(str_exp)
str_in=str_exp + '/IntermediateData/AgeCorrectedLongTADPOLE.csv'

D = pd.read_csv(str_in)
D1= D[['EXAMDATE','AGE','PTGENDER','PTEDUCAT','APOE4']].copy()
D = D.drop(['EXAMDATE','AGE','PTGENDER','PTEDUCAT','APOE4'],axis=1)
Y = D['Diagnosis'].copy()
RID = D['RID'].copy();
D = D.drop(['Diagnosis','RID'],axis=1)
D['PastDiagnosis'] = np.nan
uRIDs = np.unique(RID)

idx_diag=np.logical_not(np.isnan(Y))
Y = Y[idx_diag]
D=D[idx_diag]
RID = RID[idx_diag]
D1 = D1[idx_diag]

for i in range(len(uRIDs)):
    idx = RID == uRIDs[i]
    Yi = Y[idx].values
    Yi[1:] = Yi[:-1]
    D.loc[idx,'PastDiagnosis'] = Yi
    
idx_first = np.isnan(D['PastDiagnosis'])
D = D[np.logical_not(idx_first)]
Y = Y[np.logical_not(idx_first)]
RID = RID[np.logical_not(idx_first)]
D1 = D1[np.logical_not(idx_first)]
h = list(D)

idx_selected = pd.read_csv(str_exp+'/IntermediateData/FeatureIndices.csv')
idx_Feats = idx_selected['x'].values

import scipy.interpolate as interpolate
#R=pd.read_csv('./Data/TADPOLE_LB4.csv')
urid = np.unique(RID);
mAUC_all=[]; bca_all=[]; O_all=[]
iter_range=[250] # Should not more than 250

#P=pd.read_csv(str_exp +'IntermediateData/PredictionMatrixCluster.csv')
P=pd.read_csv(str_exp +'/IntermediateData/PredictionMatrix.csv')
Stest=P['DiseaseState']
h=list(P)
P1=P[h[:8]]
P = P[h[8:]]
S=pd.read_csv(str_exp+'/IntermediateData/PatientStages.csv',header=None)
S = S[idx_diag]
for i1 in iter_range:
    Dt= P.as_matrix()
    Dt = Dt[:,:i1]
    idx_Feats_sel=idx_Feats[:i1]
    idx_Feats_sel = np.append(idx_Feats_sel,D.shape[1]-1)
    Dtrain = D.iloc[:,idx_Feats_sel].copy()
    for j1 in range(len(urid)):
        idx_rid=RID==urid[j1]
        dtr=Dtrain[idx_rid].values
        for k1 in range(dtr.shape[1]):
            if np.sum(np.isnan(dtr[:,k1])) < len(dtr[:,k1]):
                idx_nan=np.isnan(dtr[:,k1])
                indices_val=np.where(np.logical_not(idx_nan))[0];
                fk1=interpolate.interp1d(indices_val,dtr[indices_val,k1],kind='zero',bounds_error=False, fill_value=(dtr[indices_val,k1][0],dtr[indices_val,k1][-1]));
                dtr[:,k1] = fk1(range(len(dtr[:,k1])));
        Dtrain.loc[idx_rid,:]=dtr;
    Dtrainmat = Dtrain.as_matrix()
    for j1 in range(Dtrain.shape[1]-1):
        print ([j1],end=',')
        idx_nan=np.isnan(Dtrainmat[:,j1])
        Sv=S[np.logical_not(idx_nan)].values
        Dv=Dtrainmat[np.logical_not(idx_nan),j1]
        Si=S[idx_nan].values
        idx_nan_val=np.where(idx_nan)[0]
        idx_nantest=np.isnan(Dt[:,j1])
        Sit=Stest[idx_nantest].values
        idx_nan_val_test=np.where(idx_nantest)[0]
        for k1 in range(Si.shape[0]):
            A=np.abs(Sv - Si[k1,0])
            idx_nearest=np.argsort(A[:,0])[:200]
            Dtrainmat[idx_nan_val[k1],j1]=np.mean(Dv[idx_nearest])
        for k1 in range(Sit.shape[0]):
            A=np.abs(Sv - Sit[k1])
            idx_nearest=np.argsort(A[:,0])[:200]
            Dt[idx_nan_val_test[k1],j1]=np.mean(Dv[idx_nearest])

    m = []
    s = []
    
    for i in range(Dtrainmat.shape[1]):
        m.append(np.nanmean(Dtrainmat[:,i]))
        s.append(np.nanstd(Dtrainmat[:,i]))
        Dtrainmat[np.isnan(Dtrainmat[:,i]),i]=m[i]
        Dtrainmat[:,i]=(Dtrainmat[:,i] - m[i])/s[i]

    
    import sklearn.svm as svm
    for a in [0]:
        clf = svm.SVC(kernel='rbf',class_weight='balanced',probability=True,decision_function_shape='ovo',random_state=42,C=2**a)
    
        clf.fit(Dtrainmat,Y)
        
        ## Create TestSet
        S = pd.read_csv(str_exp + '/IntermediateData/ToPredict.csv',header=None)
        S=S.values
        from openpyxl import load_workbook
        Dt1 = np.zeros((Dt.shape[0],Dt.shape[1]+1))
        Dt1[:,:-1]=Dt;
        for i in range(len(S)):
            idx_S=RID.values==S[i]
            if np.sum(idx_S)==0:
                last_diagnosis=1;
            else:
                last_diagnosis=Y.values[np.where(idx_S)[0][-1]]
            idx_S = P1['RID'].values == S[i]
            Dt1[idx_S,-1]=last_diagnosis
        
        for i in range(Dt1.shape[1]):
            Dt1[np.isnan(Dt1[:,i]),i]=m[i]
            Dt1[:,i]=(Dt1[:,i] - m[i])/s[i]
        p1 = clf.predict_proba(Dt1)
        P1 = pd.DataFrame(p1)
        P1.to_csv(str_exp + '/IntermediateData/probabilities.csv',index=False)