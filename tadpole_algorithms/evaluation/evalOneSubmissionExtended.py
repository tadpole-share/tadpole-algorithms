import numpy as np
from datetime import datetime
from evaluation import MAUC
from sklearn.metrics import confusion_matrix


def calcBCA(estimLabels, trueLabels, nrClasses):

  # Balanced Classification Accuracy

  bcaAll = []
  for c0 in range(nrClasses):
    for c1 in range(c0+1,nrClasses):
      # c0 = positive class  &  c1 = negative class
      TP = np.sum((estimLabels == c0) & (trueLabels == c0))
      TN = np.sum((estimLabels == c1) & (trueLabels == c1))
      FP = np.sum((estimLabels == c1) & (trueLabels == c0))
      FN = np.sum((estimLabels == c0) & (trueLabels == c1))

      # sometimes the sensitivity of specificity can be NaN, if the user doesn't forecast one of the classes.
      # In this case we assume a default value for sensitivity/specificity
      if (TP+FN) == 0:
        sensitivity = 0.5
      else:
        sensitivity = (TP*1.)/(TP+FN)

      if (TN+FP) == 0:
        specificity = 0.5
      else:
        specificity = (TN*1.)/(TN+FP)

      bcaCurr = 0.5*(sensitivity+specificity)
      bcaAll += [bcaCurr]
      # print('bcaCurr %f TP %f TN %f FP %f FN %f' % (bcaCurr, TP, TN, FP, FN))

  return np.mean(bcaAll)

def parseData(d4Df, forecastDf, diagLabels):

  trueDiag = d4Df['Diagnosis']
  trueADAS = d4Df['ADAS13']
  trueVents = d4Df['Ventricles']

  nrSubj = d4Df.shape[0]
  
  zipTrueLabelAndProbs = []

  hardEstimClass = -1 * np.ones(nrSubj, int)
  adasEstim = -1 * np.ones(nrSubj, float)
  adasEstimLo = -1 * np.ones(nrSubj, float)  # lower margin
  adasEstimUp = -1 * np.ones(nrSubj, float)  # upper margin
  ventriclesEstim = -1 * np.ones(nrSubj, float)
  ventriclesEstimLo = -1 * np.ones(nrSubj, float)  # lower margin
  ventriclesEstimUp = -1 * np.ones(nrSubj, float)  # upper margin

  # print('subDf.keys()', forecastDf['Forecast Date'])
  invalidResultReturn = (None,None,None,None,None,None,None,None,None,None,None)
  invalidFlag = False
  # for each subject in D4 match the closest user forecasts
  for s in range(nrSubj):
    currSubjMask = d4Df['RID'].iloc[s] == forecastDf['RID']
    currSubjData = forecastDf[currSubjMask]

    # if subject is missing
    if currSubjData.shape[0] == 0:
      print('WARNING: Subject RID %s missing from user forecasts' % d4Df['RID'].iloc[s])
      invalidFlag = True
      continue

    # if not all forecast months are present
    if currSubjData.shape[0] < 5*12: # check if at least 5 years worth of forecasts exist
      print('WARNING: Missing forecast months for subject with RID %s' % d4Df['RID'].iloc[s])
      invalidFlag = True
      continue

    currSubjData = currSubjData.reset_index(drop=True)

    timeDiffsScanCog = [d4Df['CognitiveAssessmentDate'].iloc[s] - d for d in currSubjData['Forecast Date']]
    # print('Forecast Date 2',currSubjData['Forecast Date'])
    indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]
    # print('timeDiffsScanMri', indexMin, timeDiffsScanMri)

    pCN = currSubjData['CN relative probability'].iloc[indexMin]
    pMCI = currSubjData['MCI relative probability'].iloc[indexMin]
    pAD = currSubjData['AD relative probability'].iloc[indexMin]

    # normalise the relative probabilities by their sum
    pSum = (pCN + pMCI + pAD)/3
    pCN /= pSum
    pMCI /= pSum
    pAD /= pSum

    hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])

    adasEstim[s] = currSubjData['ADAS13'].iloc[indexMin]
    adasEstimLo[s] = currSubjData['ADAS13 50% CI lower'].iloc[indexMin]
    adasEstimUp[s] = currSubjData['ADAS13 50% CI upper'].iloc[indexMin]

    # for the mri scan find the forecast closest to the scan date,
    # which might be different from the cognitive assessment date
    timeDiffsScanMri = [d4Df['ScanDate'].iloc[s] - d for d in currSubjData['Forecast Date']]
    indexMinMri = np.argsort(np.abs(timeDiffsScanMri))[0]

    ventriclesEstim[s] = currSubjData['Ventricles_ICV'].iloc[indexMinMri]
    ventriclesEstimLo[s] = currSubjData['Ventricles_ICV 50% CI lower'].iloc[indexMinMri]
    ventriclesEstimUp[s] = currSubjData['Ventricles_ICV 50% CI upper'].iloc[indexMinMri]
    # print('%d probs' % d4Df['RID'].iloc[s], pCN, pMCI, pAD)

    if not np.isnan(trueDiag.iloc[s]):
      zipTrueLabelAndProbs += [(trueDiag.iloc[s], [pCN, pMCI, pAD])]


  if invalidFlag:
    # if at least one subject was missing or if
    raise ValueError('Submission was incomplete. Please resubmit')

  # If there are NaNs in D4, filter out them along with the corresponding user forecasts
  # This can happen if rollover subjects don't come for visit in ADNI3.
  notNanMaskDiag = np.logical_not(np.isnan(trueDiag))
  trueDiagFilt = trueDiag[notNanMaskDiag]
  hardEstimClassFilt = hardEstimClass[notNanMaskDiag]

  notNanMaskADAS = np.logical_not(np.isnan(trueADAS))
  trueADASFilt = trueADAS[notNanMaskADAS]
  adasEstim = adasEstim[notNanMaskADAS]
  adasEstimLo = adasEstimLo[notNanMaskADAS]
  adasEstimUp = adasEstimUp[notNanMaskADAS]

  notNanMaskVents = np.logical_not(np.isnan(trueVents))
  trueVentsFilt = trueVents[notNanMaskVents]
  ventriclesEstim = ventriclesEstim[notNanMaskVents]
  ventriclesEstimLo = ventriclesEstimLo[notNanMaskVents]
  ventriclesEstimUp = ventriclesEstimUp[notNanMaskVents]

  assert trueDiagFilt.shape[0] == hardEstimClassFilt.shape[0]
  assert trueADASFilt.shape[0] == adasEstim.shape[0] == adasEstimLo.shape[0] == adasEstimUp.shape[0]
  assert trueVentsFilt.shape[0] == ventriclesEstim.shape[0] == \
         ventriclesEstimLo.shape[0] == ventriclesEstimUp.shape[0]

  return zipTrueLabelAndProbs, hardEstimClassFilt, adasEstim, adasEstimLo, adasEstimUp, \
    ventriclesEstim, ventriclesEstimLo, ventriclesEstimUp, trueDiagFilt, trueADASFilt, trueVentsFilt



def evalOneSub(d4Df, forecastDf):
  """
    Evaluates one submission.

  Parameters
  ----------
  d4Df - Pandas data frame containing the D4 dataset
  subDf - Pandas data frame containing user forecasts for D2 subjects.

  Returns
  -------
  mAUC - multiclass Area Under Curve
  bca - balanced classification accuracy
  adasMAE - ADAS13 Mean Aboslute Error
  ventsMAE - Ventricles Mean Aboslute Error
  adasCovProb - ADAS13 Coverage Probability for 50% confidence interval
  ventsCovProb - Ventricles Coverage Probability for 50% confidence interval

  """

  forecastDf['Forecast Date'] = [datetime.strptime(x, '%Y-%m') for x in forecastDf['Forecast Date']] # considers every month estimate to be the actual first day 2017-01
  if isinstance(d4Df['Diagnosis'].iloc[0], str):
    d4Df['CognitiveAssessmentDate'] = [datetime.strptime(x, '%Y-%m-%d') for x in d4Df['CognitiveAssessmentDate']]
    d4Df['ScanDate'] = [datetime.strptime(x, '%Y-%m-%d') for x in d4Df['ScanDate']]
    mapping = {'CN' : 0, 'MCI' : 1, 'AD' : 2}
    d4Df.replace({'Diagnosis':mapping}, inplace=True)

  diagLabels = ['CN', 'MCI', 'AD']

  zipTrueLabelAndProbs, hardEstimClass, adasEstim, adasEstimLo, adasEstimUp, \
      ventriclesEstim, ventriclesEstimLo, ventriclesEstimUp, trueDiagFilt, trueADASFilt, trueVentsFilt = \
    parseData(d4Df, forecastDf, diagLabels)
  zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)

  ########## compute metrics for the clinical status #############

  ##### Multiclass AUC (mAUC) #####

  nrClasses = len(diagLabels)
  mAUC = MAUC.MAUC(zipTrueLabelAndProbs, num_classes=nrClasses)

  ### Balanced Classification Accuracy (BCA) ###
  # print('hardEstimClass', np.unique(hardEstimClass), hardEstimClass)
  trueDiagFilt = trueDiagFilt.astype(int)
  # print('trueDiagFilt', np.unique(trueDiagFilt), trueDiagFilt)
  bca = calcBCA(hardEstimClass, trueDiagFilt, nrClasses=nrClasses)


  ## Confusion matrix ## Added by Esther Bron
  conf = confusion_matrix(hardEstimClass, trueDiagFilt.values, [0, 1, 2])
  conf = np.transpose(conf) # Transposed to match confusion matrix on web site

  print (conf)

  ####### compute metrics for Ventricles and ADAS13 ##########

  #### Mean Absolute Error (MAE) #####

  adasMAE = np.mean(np.abs(adasEstim - trueADASFilt))
  ventsMAE = np.mean(np.abs(ventriclesEstim - trueVentsFilt))

  ##### Weighted Error Score (WES) ####
  adasCoeffs = 1/(adasEstimUp - adasEstimLo)
  adasWES = np.sum(adasCoeffs * np.abs(adasEstim - trueADASFilt))/np.sum(adasCoeffs)

  ventsCoeffs = 1/(ventriclesEstimUp - ventriclesEstimLo)
  ventsWES = np.sum(ventsCoeffs * np.abs(ventriclesEstim - trueVentsFilt))/np.sum(ventsCoeffs)

  #### Coverage Probability Accuracy (CPA) ####

  adasCovProb = ( np.sum((adasEstimLo < trueADASFilt) &
                       (adasEstimUp > trueADASFilt)) * 1. )/trueADASFilt.shape[0] 
  adasCPA = np.abs(adasCovProb - 0.5)

  ventsCovProb = (np.sum((ventriclesEstimLo < trueVentsFilt) &
                        (ventriclesEstimUp > trueVentsFilt)) * 1. )/trueVentsFilt.shape[0] 
  ventsCPA = np.abs(ventsCovProb - 0.5)

  return mAUC, bca, adasMAE, ventsMAE, adasWES, ventsWES, adasCPA, ventsCPA, adasEstim, trueADASFilt
