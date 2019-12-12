"""Simplified version of evalOneSubmissionExtended.py"""
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from evaluation import MAUC
import argparse
from sklearn.metrics import confusion_matrix

from tadpole.transformations import convert_to_year_month, \
    convert_to_year_month_day, map_string_diagnosis
from tadpole.metrics import mean_abs_error, weighted_error_score, cov_prob_acc


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
  """Data preprocessing.

    Args:
        d4Df (pandas DataFrame): DataFrame containing the ground truth.
        forecastDf (pandas DataFrame): DataFrame containing the predictions.
        diagLabels (list of strings): A list of diagnosis labels.
  """

  trueDiag = d4Df['Diagnosis']
  trueADAS = d4Df['ADAS13']
  trueVents = d4Df['Ventricles']

  nrSubj = d4Df.shape[0]
  print('Number of subjects:', nrSubj)

  zipTrueLabelAndProbs = []

  hardEstimClass = -1 * np.ones(nrSubj, int)
  adasEstim = -1 * np.ones(nrSubj, float)
  adasEstimLo = -1 * np.ones(nrSubj, float)  # lower margin
  adasEstimUp = -1 * np.ones(nrSubj, float)  # upper margin
  ventriclesEstim = -1 * np.ones(nrSubj, float)
  ventriclesEstimLo = -1 * np.ones(nrSubj, float)  # lower margin
  ventriclesEstimUp = -1 * np.ones(nrSubj, float)  # upper margin

  # print('subDf.keys()', forecastDf['Forecast Date'])

  # for each subject in D4 match the closest user forecasts
  for s, (rid, currSubjInpData) in enumerate(d4Df.groupby('RID')):
    currSubjPred = forecastDf.query(f'RID == {rid}').copy()
    currSubjPred = currSubjPred.reset_index(drop=True)

    msg = None
    # if subject is missing
    if currSubjPred.shape[0] == 0:
      msg = 'Subject RID %s missing from user forecasts' % currSubjInpData['RID']
    # if not all forecast months are present
    elif currSubjPred.shape[0] < 5*12: # check if at least 5 years worth of forecasts exist
      msg = 'Missing forecast months for subject with RID %s' % currSubjInpData['RID']

    if msg is not None:
        raise ValueError(msg + '\n\nSubmission was incomplete. Please resubmit')

    indexMin = (currSubjPred['Forecast Date'] - currSubjInpData['CognitiveAssessmentDate']).idxmin()

    pCN = currSubjPred['CN relative probability'].iloc[indexMin]
    pMCI = currSubjPred['MCI relative probability'].iloc[indexMin]
    pAD = currSubjPred['AD relative probability'].iloc[indexMin]

    # normalise the relative probabilities by their sum
    pSum = (pCN + pMCI + pAD)/3
    pCN /= pSum
    pMCI /= pSum
    pAD /= pSum

    hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])

    adasEstim[s] = currSubjPred['ADAS13'].iloc[indexMin]
    adasEstimLo[s] = currSubjPred['ADAS13 50% CI lower'].iloc[indexMin]
    adasEstimUp[s] = currSubjPred['ADAS13 50% CI upper'].iloc[indexMin]

    # for the mri scan find the forecast closest to the scan date,
    # which might be different from the cognitive assessment date
    indexMinMri = (currSubjPred['Forecast Date'] - currSubjInpData['ScanDate']).idxmin()

    ventriclesEstim[s] = currSubjPred['Ventricles_ICV'].iloc[indexMinMri]
    ventriclesEstimLo[s] = currSubjPred['Ventricles_ICV 50% CI lower'].iloc[indexMinMri]
    ventriclesEstimUp[s] = currSubjPred['Ventricles_ICV 50% CI upper'].iloc[indexMinMri]
    # print('%d probs' % d4Df['RID'].iloc[s], pCN, pMCI, pAD)

    if not np.isnan(trueDiag.iloc[s]):
      zipTrueLabelAndProbs.append((trueDiag.iloc[s], [pCN, pMCI, pAD]))

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
  forecastDf - Pandas data frame containing user forecasts for D2 subjects.

  Returns
  -------
  mAUC - multiclass Area Under Curve
  bca - balanced classification accuracy
  adasMAE - ADAS13 Mean Aboslute Error
  ventsMAE - Ventricles Mean Aboslute Error
  adasCovProb - ADAS13 Coverage Probability for 50% confidence interval
  ventsCovProb - Ventricles Coverage Probability for 50% confidence interval

  """

  forecastDf['Forecast Date'] = convert_to_year_month(forecastDf['Forecast Date']) # considers every month estimate to be the actual first day 2017-01
  d4Df['CognitiveAssessmentDate'] = convert_to_year_month_day(d4Df['CognitiveAssessmentDate'])
  d4Df['ScanDate'] = convert_to_year_month_day(d4Df['ScanDate'])
  d4Df['Diagnosis'] = map_string_diagnosis(d4Df['Diagnosis'])

  diagLabels = ['CN', 'MCI', 'AD']

  zipTrueLabelAndProbs, hardEstimClass, adasEstim, adasEstimLo, adasEstimUp, \
      ventriclesEstim, ventriclesEstimLo, ventriclesEstimUp, trueDiagFilt, trueADASFilt, trueVentsFilt = \
    parseData(d4Df, forecastDf, diagLabels)
  zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
  print(len(zipTrueLabelAndProbs), zipTrueLabelAndProbs[0])

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
  # ? conf is not used, so this can be deleted?
  conf = confusion_matrix(hardEstimClass, trueDiagFilt.values, [0, 1, 2])
  conf = np.transpose(conf) # Transposed to match confusion matrix on web site

  print (conf)

  ####### compute metrics for Ventricles and ADAS13 ##########

  #### Mean Absolute Error (MAE) #####
  adasMAE = mean_abs_error(adasEstim, trueADASFilt)
  ventsMAE = mean_abs_error(ventriclesEstim, trueVentsFilt)

  ##### Weighted Error Score (WES) ####
  adasWES = weighted_error_score(adasEstim, adasEstimUp, adasEstimLo,
                                 trueADASFilt)
  ventsWES = weighted_error_score(ventriclesEstim, ventriclesEstimUp,
                                  ventriclesEstimLo, trueVentsFilt)

  #### Coverage Probability Accuracy (CPA) ####
  adasCPA = cov_prob_acc(adasEstimUp, adasEstimLo, trueADASFilt)
  ventsCPA = cov_prob_acc(ventriclesEstimUp, ventriclesEstimLo, trueADASFilt)

  return mAUC, bca, adasMAE, ventsMAE, adasWES, ventsWES, adasCPA, ventsCPA, adasEstim, trueADASFilt
