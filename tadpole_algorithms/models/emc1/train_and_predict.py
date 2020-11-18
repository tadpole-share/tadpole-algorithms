## Set Config file before use ##
import os
import warnings
def main(train_df, test_df,IntermediateDataFolder,n_boot):
    
    import subprocess
    from tadpole_algorithms.models.emc1 import EstimateDiseaseState
    str_exp=os.path.dirname(os.path.realpath(__file__))
    os.chdir(str_exp)
    import pandas as pd
    import numpy as np

    set_intermediate_results_folder(IntermediateDataFolder)

    f = open("intermediatedata.path", "r")
    IntermediateFolder = f.read()
    f.close()
    warnings.filterwarnings("ignore")

    from tadpole_algorithms.models.emc1 import PrepareData
    ## Data Preparation
    print ('Preparing Data.')
    PrepareData.main(train_df,test_df)
    print ('Step 1 / 6 Complete. Selecting Features.')
    ## Feature Selection
    subprocess.call ("/usr/bin/Rscript --vanilla ./SelectFeatures.R", shell=True)
    print ('Step 2 / 6 Complete. Training DEBM.')
    ## Estimate Disease State with DEBM
    EstimateDiseaseState.main(0)
    print ('Step 3 / 6 Complete. Predicting feature values at future timepoints.')
    ## Predict Features at Future Timepoints and Smoothen the Post Timepoints
    os.chdir(str_exp)
    subprocess.call ("/usr/bin/Rscript --vanilla ./PredictFeatures.R", shell=True)
    print ('Step 4 / 6 Complete. Training SVM classifier.')
    os.system('python3 Classify.py')
    print ('Step 5 / 6 Complete. Predicting ADAS and Ventricle values.')     
    subprocess.call ("/usr/bin/Rscript --vanilla ./PredictADASVentricles.R", shell=True)
    if n_boot==0:
        print ('Step 6 / 6 Complete. Preparing output.')
    else:
        print ('Step 6 / 6 Complete. Computing confidence intervals with bootstrapping.')
    
    P1 = pd.read_csv(IntermediateFolder+'/probabilities.csv',header=None)
    p1 = P1.values
    X=pd.DataFrame()

    ToP = pd.read_csv(IntermediateFolder+'/ToPredict.csv',header=None)

    X['RID'] = np.zeros(60*ToP.shape[0])
    
    count=-1;
    for i in range(0,(60*ToP.shape[0]),60):
        count=count+1
        X.loc[i:i+60,'RID'] = ToP.values[count,0]

    X['Forecast Date'] = ' '
    y = [2018,2019,2020,2021,2022]
    count = -1
    for k in range(ToP.shape[0]):
        for j in range(5):
            for i in range(12):
                count = count+1
                X.loc[count,'Forecast Date'] = str('%d-%02d'%(y[j],i+1))
    X['CN relative probability']=p1[:,0]
    X['MCI relative probability']=p1[:,1]
    X['AD relative probability']=p1[:,2]
    
    O1 = pd.read_csv(IntermediateFolder+'/TADPOLE_Submission_EMC1.csv')
    X['ADAS13'] = O1['ADAS13'].copy()
    X['Ventricles_ICV']=O1['Ventricles_ICV'].copy()
    
    idx_nanv = np.isnan(X['Ventricles_ICV'])
    P=pd.read_csv(IntermediateFolder +'/PredictionMatrix.csv')
    
    SP = P.loc[idx_nanv,'DiseaseState']
    str_in=IntermediateFolder + '/AgeCorrectedLongTADPOLE.csv'
    D = pd.read_csv(str_in)
    Dv=D['Ventricles']/D['ICV_bl']
    S=pd.read_csv(IntermediateFolder+'/PatientStages.csv',header=None)
    Y = D['Diagnosis'].copy()
    idx_diag=np.logical_not(np.isnan(Y))
    
    S = S[idx_diag]
    v_impute = np.zeros(SP.shape[0])
    for i in range(SP.shape[0]):
        A=np.abs(S.values - SP.values[i])
        idx_nearest=np.argsort(A[:,0])[:200]
        v_impute[i]=np.nanmean(Dv.values[idx_nearest])
    
    X.loc[idx_nanv,'Ventricles_ICV'] = v_impute
    if n_boot==0:
        X['ADAS13 50% CI lower'] = np.nan
        X['ADAS13 50% CI upper'] = np.nan
        X['Ventricles_ICV 50% CI lower'] = np.nan 
        X['Ventricles_ICV 50% CI upper'] = np.nan
    else:
        EstimateDiseaseState.main(n_boot)
        subprocess.call ("/usr/bin/Rscript --vanilla ./PredictADASVentricles_bootstrap.R", shell=True)
        xx1=np.zeros((O1.shape[0],n_boot))
        xx2=np.zeros((O1.shape[0],n_boot))
        for i in range(n_boot):
            O2 = pd.read_csv(IntermediateFolder + '/TADPOLE_Submission_EMC1_'+str(i)+'.csv')
            xx1[:,i] = O2['ADAS13'].values
            xx2[:,i] = O2['Ventricles_ICV'].values
        se1=np.nanstd(xx1,axis=1)
        se2=np.nanstd(xx2,axis=1)

        se1[np.isnan(se1)]=np.nanmean(se1)
        se2[np.isnan(se2)]=np.nanmean(se2)

        X['ADAS13 50% CI lower'] = X['ADAS13'] - 0.674*se1
        X['ADAS13 50% CI upper'] = X['ADAS13'] + 0.674*se1
        X['Ventricles_ICV 50% CI lower'] = X['Ventricles_ICV'] - 0.674*se2
        X['Ventricles_ICV 50% CI upper'] = X['Ventricles_ICV'] + 0.674*se2
    return X

def set_intermediate_results_folder(IntermediateDataFolder):

    if os.path.isdir(IntermediateDataFolder)==False:
        os.makedirs(IntermediateDataFolder)
    str_exp=os.path.dirname(os.path.realpath(__file__))
    os.chdir(str_exp)
    f = open("intermediatedata.path", "w")
    f.write(IntermediateDataFolder)
    f.close()

    return

def set_tadpole_data_folder(TadpoleFolder):
    
    f = open("tadpoledata.path", "w")
    f.write(TadpoleFolder)
    f.close() 

    return