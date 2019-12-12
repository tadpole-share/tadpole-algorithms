import os
str_exp=os.path.dirname(os.path.realpath(__file__))
os.chdir(str_exp)


text_file=open(str_exp + '/IntermediateData/pyebm.path',"r")
str_debm = text_file.read()
text_file.close()

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate

str_in=str_exp + '/IntermediateData/AgeCorrectedLongTADPOLE.csv'

D = pd.read_csv(str_in)
D1= D[['EXAMDATE','AGE','PTGENDER','PTEDUCAT','APOE4']].copy()
D = D.drop(['EXAMDATE','AGE','PTGENDER','PTEDUCAT','APOE4'],axis=1)
Y = D['Diagnosis'].copy()
RID = D['RID'].copy();
D = D.drop(['Diagnosis','RID'],axis=1)
uRIDs = np.unique(RID)

Dtestfull = D.copy()
RIDtest = RID.copy()

idx_diag=np.logical_not(np.isnan(Y))
Y = Y[idx_diag]
D=D[idx_diag]
RID = RID[idx_diag]
D1 = D1[idx_diag]

idx_selected = pd.read_csv(str_exp+'/IntermediateData/FeatureIndices.csv')
idx_Feats = idx_selected['x'].values

#R=pd.read_csv('./Data/TADPOLE_LB4.csv')
urid = np.unique(RID);
mAUC_all=[]; bca_all=[]
iter_range=[250]

V=np.isnan(D.values)
perc_missing = (np.sum(V,axis=0)*1. / V.shape[0])
for i1 in iter_range:
    idx_Feats_sel=idx_Feats[:i1]
    idx_nonmissing_sel = perc_missing[idx_Feats_sel] < 0.20
    idx_Feats_sel = idx_Feats_sel[idx_nonmissing_sel]
                 
    Dtrain = D.iloc[:,idx_Feats_sel].copy()    
    Dtest = Dtestfull.iloc[:,idx_Feats_sel].copy()    
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
    for j1 in range(len(urid)):
        idx_rid_test = RIDtest == urid[j1]
        dtrtest=Dtest[idx_rid_test].values
        for k1 in range(dtrtest.shape[1]):
            if np.sum(np.isnan(dtrtest[:,k1])) < len(dtrtest[:,k1]):
                idx_nan=np.isnan(dtrtest[:,k1])
                indices_val=np.where(np.logical_not(idx_nan))[0];
                fk1=interpolate.interp1d(indices_val,dtrtest[indices_val,k1],kind='zero',bounds_error=False, fill_value=(dtrtest[indices_val,k1][0],dtrtest[indices_val,k1][-1]));
                dtrtest[:,k1] = fk1(range(len(dtrtest[:,k1])));
        Dtest.loc[idx_rid_test,:]=dtrtest;
    Ystr = Y.copy()
    Ystr[np.isnan(Ystr)]=2
    Ystr[Ystr == 1] = 'CN'; Ystr[Ystr == 2] = 'MCI'; Ystr[Ystr == 3] = 'AD'; 
    Dtrain['Diagnosis']=Ystr
    Dtrain['PTID'] = RID
    Dtest['PTID'] = RIDtest
    Dtest['Diagnosis']='CN'
    #Dtrain['APOE4'] = D1['APOE4']
    os.chdir(str_debm)
    from pyebm import debm
    #import LDEBM
    from collections import namedtuple
    MO = namedtuple('MethodOptions','MixtureModel Bootstrap PatientStaging');
    MO.Bootstrap=10; MO.MixtureModel='GMMvv2'; MO.PatientStaging=['exp','p'];
    VO = namedtuple('VerboseOptions','Distributions' 'PatientStaging');
    VO.Distributions=0; VO.PatientStaging=0;
    ModelOutput,SubjTrainAll,SubjTestAll=debm.fit(Dtrain,Factors=[],MethodOptions=MO,VerboseOptions=VO,DataTest=Dtest)
    #SubjectInfo,pi0_all,event_centers_all=LDEBM.Clustering(SubjTrainAll[0].copy(),params_opt=ModelOutput.BiomarkerParameters[0])
    #print [ModelOutput.BiomarkerList[x] for x in ModelOutput.MeanCentralOrdering]

    
ST=SubjTrainAll[0]
#S=ST['Stages']
#dS=np.zeros(S.shape)
#for j1 in range(len(urid)):
#    idx_rid=RID==urid[j1]
#    if np.sum(idx_rid)>1:
#        dS[idx_rid]=np.gradient(S.values[idx_rid]) / np.gradient(D1.loc[idx_rid,'AGE'])
#dS[np.isinf(dS)]=0

#ST['Rate']=dS
for i in range(10):
    SubjTestAll[i]['Stages'].to_csv(str_exp+'IntermediateData/PatientStages_'+str(i)+'.csv',index=False)
