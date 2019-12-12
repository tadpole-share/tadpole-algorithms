# Prepare data script for longitudinal predictions.
import os
import pandas as pd
import numpy as np

from tqdm.auto import tqdm


def preprocess(tadpoleD1D2File, out_dir, leaderboard=0, tadpoleLB1LB2File=''):

    # Settings
    #leaderboard = 0

    # Input directory
    #str_exp = os.path.dirname(os.path.realpath(__file__))
    #os.chdir(str_exp)

    # Input file
    #tadpoleD1D2File = os.path.join(str_exp, 'Data', 'TADPOLE_D1_D2.csv')
    Dtadpole = pd.read_csv(tadpoleD1D2File)

    idx_progress = np.logical_and(Dtadpole['DXCHANGE'] >= 4, Dtadpole['DXCHANGE'] <= 6)
    SubC = np.unique(Dtadpole.loc[idx_progress, 'RID'])
    SubC = pd.Series(SubC)
    SubC.to_csv(os.path.join(out_dir, 'IntermediateData', 'SubjectsWithChange.csv'), index=False)

    # Recode diagnosis
    idx_mci = Dtadpole['DXCHANGE'] == 4
    Dtadpole.loc[idx_mci, 'DXCHANGE'] = 2
    idx_ad = Dtadpole['DXCHANGE'] == 5
    Dtadpole.loc[idx_ad, 'DXCHANGE'] = 3
    idx_ad = Dtadpole['DXCHANGE'] == 6
    Dtadpole.loc[idx_ad, 'DXCHANGE'] = 3
    idx_cn = Dtadpole['DXCHANGE'] == 7
    Dtadpole.loc[idx_cn, 'DXCHANGE'] = 1
    idx_mci = Dtadpole['DXCHANGE'] == 8
    Dtadpole.loc[idx_mci, 'DXCHANGE'] = 2
    idx_cn = Dtadpole['DXCHANGE'] == 9
    Dtadpole.loc[idx_cn, 'DXCHANGE'] = 1
    Dtadpole = Dtadpole.rename(columns={'DXCHANGE': 'Diagnosis'})
    h = list(Dtadpole)
    Dtadpole['AGE'] += Dtadpole['Month_bl'] / 12.

    D2 = Dtadpole['D2'].copy()
    Dtadpole = Dtadpole.drop(h[1:8]+[h[9]]+h[14:17]+h[45:47]+h[53:73]+h[74:486]+h[832:838]+h[1172:1174]+h[1657:1667]+h[1895:1902]+h[1905:], 1)

    h = list(Dtadpole)

    print('Forcing Numeric Values')
    for i in tqdm(range(5, len(h))):
        if Dtadpole[h[i]].dtype != 'float64':
            Dtadpole[h[i]] = pd.to_numeric(Dtadpole[h[i]], errors='coerce')

    urid = np.unique(Dtadpole['RID'].values)

    Dtadpole_sorted = pd.DataFrame(columns=h)
    print('Sort the dataframe based on age for each subject')
    for i in tqdm(range(len(urid))):
        agei = Dtadpole.loc[Dtadpole['RID'] == urid[i], 'AGE']
        idx_sortedi = np.argsort(agei)
        D1 = Dtadpole.loc[idx_sortedi.index[idx_sortedi]]
        ld = [Dtadpole_sorted, D1]
        Dtadpole_sorted = pd.concat(ld)

    if not os.path.exists(os.path.join(out_dir, 'IntermediateData')):
        os.mkdir(os.path.join(out_dir, 'IntermediateData'))
    Dtadpole_sorted.to_csv(os.path.join(out_dir, 'IntermediateData', 'LongTADPOLE.csv'), index=False)

    if leaderboard:
        #tadpoleLB1LB2File = os.path.join(out_dir, 'Data', 'TADPOLE_LB1_LB2.csv')
        LB_Table = pd.read_csv(tadpoleLB1LB2File)
        LB = LB_Table['LB1']+LB_Table['LB2']
        idx_lb = LB.values >= 1
        Dtadpole = Dtadpole[idx_lb]

        # Leaderboard
        idx_lb2 = LB_Table['LB2'] == 1
        LB2_RID = LB_Table.loc[idx_lb2, 'RID']
        SLB2 = pd.Series(np.unique(LB2_RID.values))
        SLB2.to_csv(os.path.join(out_dir, 'IntermediateData', 'ToPredict.csv'), index=False)

    else:
        # Submission
        idx_d2 = D2 == 1
        Dtadpole_RID = Dtadpole.loc[idx_d2, 'RID']
        SD2 = pd.Series(np.unique(Dtadpole_RID.values))
        SD2.to_csv(os.path.join(out_dir, 'IntermediateData', 'ToPredict_D2.csv'), index=False)
