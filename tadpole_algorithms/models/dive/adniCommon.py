import nibabel as nib
import numpy as np
import scipy
import time
import sys
import pickle
import gc
from matplotlib import pyplot as pl

from env import *
import evaluationFramework

def testMeanBiomkValue(data, diag, pointIndices, plotTrajParams):
  fsaverageAnnotFile = '%s/subjects/fsaverage/label/lh.aparc.annot' % plotTrajParams['freesurfPath']
  labels, ctab, names = nib.freesurfer.io.read_annot(fsaverageAnnotFile, orig_ids = False)

  # print(labels, ctab, names)
  # print(labels.shape, ctab.shape, len(names))
  # print(data.shape, diag.shape, labels.shape[0])
  assert(data.shape[0] == diag.shape[0])

  unqLabels = np.sort(np.unique(labels))
  nrLabels = len(unqLabels)

  # print(diag)
  diagNrs = np.sort(np.unique(diag))
  nrDiag = diagNrs.shape[0]

  means = np.zeros((nrDiag, nrLabels), float)
  stds = np.zeros((nrDiag, nrLabels), float)

  # print(list(enumerate(names)))
  #print(adsa)
  entLabelNr = 6

  # for d in range(nrDiag):
  #   for l in range(nrLabels):
  #     #print('nr Vertices in region %s is %d' % ( names[l], np.sum(labels == unqLabels[l] )))
  #     print(data.shape, labels[pointIndices].shape)
  #     tmpData = data[diag == diagNrs[d],:][:,labels[pointIndices] == unqLabels[l]]
  #     #print(tmpData)
  #     means[d,l] = np.mean(tmpData, axis=(0,1))
  #     stds[d,l] = np.std(np.mean(tmpData, axis = 1)) # mean across points, std across subjects
  #
  #   print('%s %s  %.3f +/- %.3f' %
  #     (names[entLabelNr].decode('utf-8'),
  #     plotTrajParams['diagLabels'][diagNrs[d]],
  #     means[d, entLabelNr], stds[d,entLabelNr]))

  # print(adasd)

  # print([(i,names[i]) for i in range(nrLabels)] )

  # def doTtest(data, diag, pointIndices):

  nrSubjCross, nrBiomk = data.shape
  #pVals = np.nan * np.ones(nrBiomk,float)

  # perform t-test on every voxel, sort them by p-values
  pVals = scipy.stats.ttest_ind(data[diag == CTL,:], data[diag == AD,:])[1]

  sortedInd = np.argsort(pVals)

  # print('pointIndices[sortedInd]', pointIndices[sortedInd][:100])
  # print('names', [names[i] for i in labels[pointIndices[sortedInd]][:100]])


  return sortedInd, labels, names

def readDataFile(inputFileData, cluster):

  if cluster:
    time.sleep(3)
    f = open(inputFileData, "rb")
    time.sleep(3)
    binary_data = f.read()  # This part doesn't take long
    print('loaded into memory')
    sys.stdout.flush()
    time.sleep(3)
    dataStruct = pickle.loads(binary_data)  # This takes ages
    print('created data struct')
    sys.stdout.flush()
    return dataStruct


  else:
    # running on local machine
    #dataStruct = pickle.load(open(inputFileData, 'rb'))

    f = open(inputFileData, "rb")
    binary_data = f.read()  # This part doesn't take long
    print('loaded into memory')
    dataStruct = pickle.loads(binary_data)  # This takes ages
    print('created data struct')

    # print(dataStruct)
    # print(adsdsa)

    return dataStruct

def printBICresults(params, expNameBefCl, expNameAfterCl, modelToRun,
  nrClustList, runAllExpFunc):

  fileName = 'resfiles/BICres_%s%s' % (expNameBefCl, expNameAfterCl)
  bicAllFileName = '%s.npz' % fileName
  figBicFileName = '%s.png' % fileName

  runPart = 'R'
  if runPart == 'R':

    bic = np.nan * np.ones(len(nrClustList), float)
    aic = np.nan * np.ones(len(nrClustList), float)

    # go through every nrClustList file that was found for this experiment
    for nrClustIndex in range(len(nrClustList)):
      nrClustCurr = nrClustList[nrClustIndex]

      expName = '%sCl%d%s' % (expNameBefCl, nrClustCurr, expNameAfterCl)
      params['plotTrajParams']['expName'] = expName

      params['nrClust'] = nrClustCurr
      # [initClust, modelFit, aic/bic, plotBlender, sampleTraj]
      params['runPartStd'] = ['Non-enforcing', 'Non-enforcing' ,
        'Non-enforcing', 'I', 'I']
      params['runPartMain'] = ['R', 'I', 'I']  # [mainPart, plot, stage]

      modelNames, res = evaluationFramework.runModels(params, expName, modelToRun, runAllExpFunc)

      if res[0]['std']:
        print('bic', res[0]['std']['bic'])
        print('aic', res[0]['std']['aic'])
        bic[nrClustIndex] = res[0]['std']['bic']
        aic[nrClustIndex] = res[0]['std']['aic']

      res = None
      gc.collect()
      print('garbage collector called')
      sys.stdout.flush()

    dataStruct = dict(aic=aic, bic=bic, nrClustList=nrClustList)
    pickle.dump(dataStruct, open(bicAllFileName, 'wb'), pickle.HIGHEST_PROTOCOL)

  elif runPart == 'L':
    dataStruct = pickle.load(open(bicAllFileName, 'rb'))
    aic = dataStruct['aic']
    bic = dataStruct['bic']
  else:
    raise ValueError('need to either load file or run the experiment')

  foundInd = np.logical_not(np.isnan(bic))
  bicFound = bic[foundInd]
  aicFound = aic[foundInd]
  nrClustListFound = np.array(nrClustList)[foundInd]

  minBICInd = np.argmin(bicFound)
  minBIC = bicFound[minBICInd]
  nrClustMinBIC = nrClustListFound[minBICInd]

  minAICInd = np.argmin(aicFound)
  minAIC = aicFound[minAICInd]
  nrClustMinAIC = nrClustListFound[minAICInd]

  print('bicFound, nrClustListBicFound', bicFound, nrClustListFound)
  print('minBIC, nrClustMinBIC', minBIC, nrClustMinBIC)

  print('aicFound, nrClustListAicFound', aicFound, nrClustListFound)
  print('minAIC, nrClustMinAIC', minAIC, nrClustMinAIC)

  fig = pl.figure()
  colors = ['r', 'g']

  pl.plot(nrClustListFound, bicFound, label='BIC', color=colors[0])
  pl.plot(nrClustListFound, aicFound, label='AIC', color=colors[1])

  # plot the two dots

  size = 50
  pl.scatter([nrClustMinBIC, nrClustMinAIC],
    [minBIC, minAIC], color=colors, s=size)

  fontsize = 14

  pl.legend(fontsize=fontsize)
  pl.xlabel('Number of clusters',fontsize=fontsize)
  pl.ylabel('Criterion Value', fontsize=fontsize)
  nrClustToPlot = 8
  pl.xlim([1,2+nrClustToPlot])

  allPlottedBicAicValues = np.concatenate((bicFound[:nrClustToPlot], aicFound[:nrClustToPlot]))
  yMax = np.max(allPlottedBicAicValues, axis=0)
  yMin = np.min(allPlottedBicAicValues, axis=0)
  yDelta = (yMax - yMin)/6

  pl.ylim([yMin-yDelta,yMax+yDelta])
  pl.xticks(range(2,3+nrClustToPlot),fontsize=fontsize)
  pl.yticks(fontsize=fontsize)

  fig.savefig(figBicFileName, dpi=100)

  return fig