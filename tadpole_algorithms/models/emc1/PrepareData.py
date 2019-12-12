import os
str_exp=os.path.dirname(os.path.realpath(__file__))
os.chdir(str_exp)

text_file=open(str_exp + '/IntermediateData/datafile.path',"r")
tadpoleD1D2File = text_file.read()
text_file.close()

import pandas as pd
import numpy as np
Dtadpole=pd.read_csv(tadpoleD1D2File)
D2=Dtadpole['D2']
I=Dtadpole['ICV_bl']
imean=np.nanmean(I)
idx_na=np.isnan(I)
urid=np.unique(Dtadpole.loc[idx_na,'RID'])
Iany=Dtadpole['ICV']
for i in range(len(urid)):
    rid=urid[i]
    idx_rid=Dtadpole['RID']==rid
    icv=Iany[idx_rid]
    icv = icv[np.logical_not(np.isnan(icv))]
    if len(icv)>0:
        Dtadpole.loc[idx_rid,'ICV_bl']=icv.values[0]
    else:
        Dtadpole.loc[idx_rid,'ICV_bl']=imean

idx_progress=np.logical_and(Dtadpole['DXCHANGE']>=4, Dtadpole['DXCHANGE']<=6)
SubC=np.unique(Dtadpole.loc[idx_progress,'RID'])
SubC = pd.Series(SubC);
SubC.to_csv(str_exp+'/IntermediateData/SubjectsWithChange.csv',index=False)

idx_mci=Dtadpole['DXCHANGE']==4
Dtadpole.loc[idx_mci,'DXCHANGE']=2
idx_ad = Dtadpole['DXCHANGE']==5
Dtadpole.loc[idx_ad,'DXCHANGE']=3
idx_ad = Dtadpole['DXCHANGE']==6
Dtadpole.loc[idx_ad,'DXCHANGE']=3
idx_cn = Dtadpole['DXCHANGE']==7
Dtadpole.loc[idx_cn,'DXCHANGE']=1
idx_mci=Dtadpole['DXCHANGE']==8
Dtadpole.loc[idx_mci,'DXCHANGE']=2
idx_cn = Dtadpole['DXCHANGE']==9
Dtadpole.loc[idx_cn,'DXCHANGE']=1
Dtadpole=Dtadpole.rename(columns={'DXCHANGE':'Diagnosis'})
h = list(Dtadpole)
Dtadpole['AGE']  += Dtadpole['Month_bl'] / 12.
        
Dtadpole=Dtadpole.drop(h[1:8]+[h[9]]+h[14:17]+h[45:47]+h[53:73]+h[74:486]+h[832:838]+h[1172:1174]+h[1657:1667]+h[1895:1902]+h[1905:],1)

h = list(Dtadpole)
#idx_nan=np.isnan(Dtadpole['Diagnosis'].values)                  
#Dtadpole = Dtadpole[np.logical_not(idx_nan)]
print ('Forcing Numeric Values')
for i in range(5,len(h)):
    print ([i],end=',')
    if Dtadpole[h[i]].dtype != 'float64':
        Dtadpole[h[i]]=pd.to_numeric(Dtadpole[h[i]], errors='coerce')

urid = np.unique(Dtadpole['RID'].values)

Dtadpole_sorted=pd.DataFrame(columns=h)
print ('Sort the dataframe based on age for each subject')
for i in range(len(urid)):
    print ([i],end=',')
    agei=Dtadpole.loc[Dtadpole['RID']==urid[i],'AGE']
    idx_sortedi=np.argsort(agei)
    D1=Dtadpole.loc[idx_sortedi.index[idx_sortedi]]
    ld = [Dtadpole_sorted,D1]
    Dtadpole_sorted = pd.concat(ld)

h = ['CDRSB','EcogSPOrgan','EcogSPVisspat','EcogSPTotal','EcogSPPlan',
     'EcogSPLang','EcogSPDivatt','FAQ','MMSE','ADAS11','ADAS13']
Dtadpole_sorted[h]=np.log(Dtadpole_sorted[h] - np.min(Dtadpole_sorted[h]) + 1.)
    
Dtadpole_sorted.to_csv(str_exp+'/IntermediateData/LongTADPOLE.csv',index=False)

idx_lb2=D2==1
LB2_RID = Dtadpole.loc[idx_lb2,'RID']
SLB2=pd.Series(np.unique(LB2_RID.values))
SLB2.to_csv(str_exp+'/IntermediateData/ToPredict.csv',index=False)
