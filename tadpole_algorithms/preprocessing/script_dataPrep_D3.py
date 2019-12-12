# Prepare data script for crosssectional predictions (D3).
import os
import pandas as pd
import numpy as np

from tqdm.auto import tqdm


def preprocess(tadpoleD3File, tadpoleD1D2File, out_dir):

    # Input directory
    #str_exp = os.path.dirname(os.path.realpath(__file__))
    #os.chdir(str_exp)

    # Input file D3
    #tadpoleD3File = os.path.join(str_exp, 'Data', 'TADPOLE_D3.csv')
    Dtadpole = pd.read_csv(tadpoleD3File)

    # Make D3 prediction file for testing
    # Recode diagnosis
    idx_mci = Dtadpole['DX'] == 'MCI'
    Dtadpole.loc[idx_mci, 'DX'] = 2
    idx_mci = Dtadpole['DX'] == 'NL to MCI'
    Dtadpole.loc[idx_mci, 'DX'] = 2
    idx_mci = Dtadpole['DX'] == 'Dementia to MCI'
    Dtadpole.loc[idx_mci, 'DX'] = 2
    idx_ad = Dtadpole['DX'] == 'Dementia'
    Dtadpole.loc[idx_ad, 'DX'] = 3
    idx_ad = Dtadpole['DX'] == 'MCI to Dementia'
    Dtadpole.loc[idx_ad, 'DX'] = 3
    idx_ad = Dtadpole['DX'] == 'NL to Dementia'
    Dtadpole.loc[idx_ad, 'DX'] = 3
    idx_cn = Dtadpole['DX'] == 'NL'
    Dtadpole.loc[idx_cn, 'DX'] = 1
    idx_cn = Dtadpole['DX'] == 'MCI to NL'
    Dtadpole.loc[idx_cn, 'DX'] = 1
    idx_cn = Dtadpole['DX'] == 'Dementia to NL'
    Dtadpole.loc[idx_cn, 'DX'] = 1
    Dtadpole = Dtadpole.rename(columns={'DX': 'Diagnosis', 'ICV': 'ICV_bl'})
    h = list(Dtadpole)

    Dtadpole = Dtadpole.drop([h[1]]+h[7:11]+h[20:37], 1)

    h = list(Dtadpole)

    print('Forcing Numeric Values')
    for i in tqdm(range(5, len(h))):
        if Dtadpole[h[i]].dtype != 'float64':
            Dtadpole[h[i]] = pd.to_numeric(Dtadpole[h[i]], errors='coerce')

    Dtadpole = Dtadpole.sort_values(['RID'])

    if not os.path.exists(os.path.join(out_dir, 'IntermediateData')):
        os.mkdir(os.path.join(out_dir, 'IntermediateData'))
    Dtadpole.to_csv(os.path.join(out_dir, 'IntermediateData', 'LongTADPOLE_D3.csv'), index=False)

    # Input file D1
    #tadpoleD1D2File = os.path.join(str_exp, 'Data', 'TADPOLE_D1_D2.csv')
    Dtadpole = pd.read_csv(tadpoleD1D2File)

    # Make D1 prediction file for training (only D1 subects that are not in D3)
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

    idx_notd2 = Dtadpole['D2'] == 0
    Dtadpole = Dtadpole[idx_notd2]

    Dtadpole = Dtadpole.drop(h[1:8]+[h[9]]+h[14:23]+h[25:47]+h[53:73]+h[74:486]+h[832:],1)

    h = list(Dtadpole)

    print('Forcing Numeric Values')
    for i in tqdm(range(5,len(h))):
        if Dtadpole[h[i]].dtype != 'float64':
            Dtadpole[h[i]]=pd.to_numeric(Dtadpole[h[i]], errors='coerce')

    urid = np.unique(Dtadpole['RID'].values)

    Dtadpole_sorted=pd.DataFrame(columns=h)
    print('Sort the dataframe based on age for each subject')
    for i in tqdm(range(len(urid))):
        agei=Dtadpole.loc[Dtadpole['RID']==urid[i],'AGE']
        idx_sortedi=np.argsort(agei)
        D1=Dtadpole.loc[idx_sortedi.index[idx_sortedi]]
        ld = [Dtadpole_sorted,D1]
        Dtadpole_sorted = pd.concat(ld)

    Dtadpole_sorted.to_csv(os.path.join(out_dir, 'IntermediateData', 'LongTADPOLE_D1.csv'), index=False)
