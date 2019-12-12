import voxelDPM
import VDPMMean
import numpy as np
import scipy
import sys
from env import *
import os
import pickle
import sklearn
import math


''' Class for a Voxelwise Disease Progression Model that can handle missing data (NaNs)'''

class VDPMNanBuilder(voxelDPM.VoxelDPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMNan(dataIndices, expName, params, self.plotterObj)

class VDPMNan(VDPMMean.VDPMMean):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)

    self.nanMask = np.nan

  def runInitClust(self, runPart, crossData, crossDiag):
    # printStats(longData, longDiag)
    sys.stdout.flush()
    np.random.seed(1)
    plotTrajParams = self.params['plotTrajParams']
    initClustFile = '%s/initClust.npz' % self.outFolder
    nrClust = self.params['nrClust']
    labels = self.params['labels']
    if runPart[0] == 'R':
      # perform some data driven clustering in order to get some initial clustering probabilities

      if self.params['initClustering'] == 'k-means':
        # perform k-means usign scikit-learn
        initClustSubsetInd = self.params['initClustSubsetInd']
        nearNeighInitClust = self.params['nearNeighInitClust']

        # if subject has nans, fill out the values with mean for that diagnosis
        dataToCluster = crossData[:,initClustSubsetInd].copy()
        if np.sum(np.isnan(dataToCluster),axis=(0,1)) > 0:
          print(np.sum(np.sum(np.isnan(dataToCluster),axis=1) == 0))
          unqDiags = np.unique(crossDiag)
          nrDiags = len(unqDiags)
          nrSubjCross, nrBiomk = crossData.shape
          meanValuesDB = np.zeros((nrDiags, nrBiomk))
          meanDataSB = np.zeros((nrSubjCross, nrBiomk))
          for d in range(nrDiags):
            print(crossData[crossDiag == unqDiags[d],:])
            print(np.nanmean(crossData[crossDiag == unqDiags[d],:],axis=0))
            meanValuesDB[d,:] = np.nanmean(crossData[crossDiag == unqDiags[d],:],axis=0)
            meanDataSB[crossDiag == unqDiags[d],:] = meanValuesDB[d,:]

          print('dataToCluster[::100,:][::100]', dataToCluster[::100,:][::100])
          dataToCluster[np.isnan(dataToCluster)] = meanDataSB[np.isnan(dataToCluster)]
          print('dataToCluster[::100,:][::100]', dataToCluster[::100, :][::100])
          # print(adsd)

        clustResStruct = sklearn.cluster.KMeans(n_clusters=nrClust, random_state=0)\
          .fit(dataToCluster.T)

        initClustSubset = clustResStruct.labels_  # indices should start from 0
        initClust = initClustSubset[nearNeighInitClust]

        print('clustHist', [np.sum(initClust == c) for c in range(nrClust)])
        print('clustHistSubset', [np.sum(initClustSubset == c) for c in range(nrClust)])
        print('initClust', initClust.shape, initClust)
        print('initClustSubset', initClustSubset.shape, initClustSubset)
        # print(ada)
        assert (np.min(initClustSubset) == 0)
        assert (np.min(initClust) == 0)

      elif self.params['initClustering'] == 'hist':
        # assumes data has been z-scored
        print(np.std(crossData[crossDiag == CTL], axis=0))
        assert (all(np.nanmean(crossData[crossDiag == CTL], axis=0) < 0.5))
        avgCrossDataPatB = np.nanmean(crossData[crossDiag == self.params['patientID']], axis=0)
        percentiles = list(np.percentile(avgCrossDataPatB,
          [c * 100 / nrClust for c in range(1, nrClust)]))
        percentiles = [-float("inf")] + percentiles + [float("inf")]


        print('avgCrossDataPatB', avgCrossDataPatB, labels[np.isnan(avgCrossDataPatB)])
        print('percentiles', percentiles)
        assert (len(percentiles) == (nrClust + 1))
        initClust = np.zeros(crossData.shape[1], int)
        for c in range(nrClust):
          clustMask = np.logical_and(percentiles[c] < avgCrossDataPatB,
            avgCrossDataPatB < percentiles[c + 1])
          initClust[clustMask] = c


      os.system('mkdir -p %s' % self.outFolder)
      clustDataStruct = dict(initClust=initClust)
      pickle.dump(clustDataStruct, open(initClustFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    elif runPart[0] == 'L':
      clustDataStruct = pickle.load(open(initClustFile, 'rb'))
      initClust = clustDataStruct['initClust']
      assert (np.min(initClust) == 0)
    else:
      print('no file found at runPart[0] for %s' % self.outFolder)
      # initClust = np.zeros(crossData.shape[1], int)
      return None

    # print('initClust', initClust)
    # for c in range(nrClust):
    #   print('clust %d' % c, self.params['labels'][initClust == c])


    return initClust


  def initTrajParams(self, crossData, crossDiag, clustProbBC, crossAgeAtScan, subShiftsLong,
                     uniquePartCodeInverse, crossAge1array, extraRangeFactor):
    ''' ATTENTION: crossData will contain NaNs!

    initialises sigmoidal trajectory params [a,b,c,d] with minimum a, maximum a+d, slope a*b/4
      and slope maximum attained at center c

    '''

    print('crossDiag', np.sum(np.isnan(crossDiag)), crossDiag)
    assert not np.isnan(crossDiag).any()
    assert not np.isnan(clustProbBC).any()
    assert not np.isnan(crossAgeAtScan).any()
    assert not np.isnan(subShiftsLong).any()
    assert not np.isnan(uniquePartCodeInverse).any()
    assert not np.isnan(crossAge1array).any()

    clustProbBCColNorm = clustProbBC / np.sum(clustProbBC, 0)[None, :]

    nrClust = clustProbBC.shape[1]
    initSubShiftsCross = subShiftsLong[uniquePartCodeInverse, :]
    variances = np.zeros(nrClust, float)
    dpsCross = VDPMNan.calcDps(initSubShiftsCross, crossAge1array)

    # calculate average voxel value for each (subject, cluster) pair, use them to initialise theta
    thetas = np.zeros((nrClust, 4), float)
    minMaxRange = np.nanmax(crossData, 0) - np.nanmin(crossData, 0)
    print('minMaxRange', minMaxRange.shape)
    print('clustProbBCColNorm', clustProbBCColNorm.shape)
    minMaxRangeC = np.dot(minMaxRange, clustProbBCColNorm)
    dataMinB = np.nanmin(crossData, 0)
    dataMinC = np.dot(dataMinB, clustProbBCColNorm)
    assert (1 <= self.params['patientID'] <= 3)

    print('crossDiag', crossDiag, 'patientID', self.params['patientID'])
    print('extraRangeFactor', extraRangeFactor)
    thetas[:, 0] = minMaxRangeC + minMaxRangeC * 2 * extraRangeFactor / 10  # ak = (dk + ak) - dk
    thetas[:, 1] = -16 / (thetas[:, 0] * np.std(dpsCross[crossDiag == self.params['patientID']]))  # bk = 4/ak so that slope = bk*ak/4 = -1
    thetas[:, 2] = np.mean(crossAgeAtScan)  # ck
    thetas[:, 3] = dataMinC - minMaxRangeC * extraRangeFactor / 10  # dk

    # print('thetas', thetas)

    thetas = self.makeThetasIdentif(thetas, shiftTransform=[0, 1])

    # print('thetas', thetas)
    # print(adsa)

    dataStdB = np.nanstd(crossData, axis=0)
    dataStdC = np.dot(dataStdB, clustProbBCColNorm)
    variances = np.power(dataStdC, 2)


    return thetas, variances

  def inferMissingData(self, crossData,longData, prevClustProbBC, thetas, subShiftsCross,
    crossAge1array, trajFunc, scanTimepts, partCode, uniquePartCode, plotterObj):

    self.nanMask = np.isnan(crossData)

    prevClustProbColNormBC = prevClustProbBC / np.sum(prevClustProbBC, 0)[None, :]
    (nrSubjCross, nrBiomk) = crossData.shape
    nrClust = thetas.shape[0]
    dps = voxelDPM.VoxelDPM.calcDps(subShiftsCross, crossAge1array)

    fSC = np.zeros((nrSubjCross, nrClust), float)
    for k in range(nrClust):
      fSC[:, k] = trajFunc(dps, thetas[k, :])

    # print('longData[s]', [longData[s] for s in range(5)])

    dataInferredSB = np.dot(fSC, prevClustProbBC.T)
    # print('crossData[:3,:]', crossData[:3,:])
    crossData[self.nanMask] = dataInferredSB[self.nanMask]
    longData = self.makeLongArray(crossData, scanTimepts, partCode, uniquePartCode)

    # print('crossData[:3,:]', crossData[:3, :])
    # print('longData[s]', [longData[s] for s in range(5)])
    # print(adss)


    return crossData, longData

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
    trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode):
    # overwrite function as we need to use a different variance (in the biomk measurements as opposed to their mean)

    prevClustProbColNormBC = prevClustProbBC / np.sum(prevClustProbBC, 0)[None, :]
    (nrSubj, nrBiomk) = crossData.shape
    nrClust = thetas.shape[0]
    dps = voxelDPM.VoxelDPM.calcDps(subShiftsCross, crossAge1array)

    fSC = np.zeros((nrSubj, nrClust), float)
    for k in range(nrClust):
      fSC[:, k] = trajFunc(dps, thetas[k, :])

    # dataInferredSB = np.dot(fSC, prevClustProbBC.T)
    # crossData[self.nanMask] = dataInferredSB[self.nanMask]

    crossData, longData = self.inferMissingData(crossData, longData, prevClustProbBC,
      thetas, subShiftsCross, crossAge1array, trajFunc, scanTimepts, partCode, uniquePartCode, self.plotterObj)

    varianceIndivBiomk = np.zeros(variances.shape, float)
    # estimate the variance in the biomk noise, as opposed to the variance in the mean
    for c in range(nrClust):
      # call super method
      finalSSD = voxelDPM.VoxelDPM.objFunTheta(self, thetas[c,:], crossData, dps,
        prevClustProbColNormBC[:,c])[1]
      varianceIndivBiomk[c] = finalSSD / (crossData.shape[0])


    logClustProb = np.zeros((nrBiomk, nrClust), float)
    clustProb = np.zeros((nrBiomk, nrClust), float)
    tmpSSD = np.zeros((nrBiomk, nrClust), float)
    tmpSSDVar = np.zeros((nrBiomk, nrClust), float)
    for k in range(nrClust):
      tmpSSD[:, k] = np.sum(np.power(crossData - fSC[:, k][:, None], 2),
                            0)  # sum across subjects, left with 1 x NR_BIOMK array
      assert (tmpSSD[:, k].shape[0] == nrBiomk)
      tmpSSDVar[:, k] = -tmpSSD[:, k] / (2 * varianceIndivBiomk[k])
      logClustProb[:, k] = -tmpSSD[:, k] / (2 * varianceIndivBiomk[k]) - np.log(2 * math.pi * varianceIndivBiomk[k]) * nrSubj / 2

    # vertexNr = 755
    # print('tmpSSD[vertexNr,:]', tmpSSD[vertexNr, :])  # good
    # print('tmpSSDVar[vertexNr,:] ', tmpSSDVar[vertexNr, :])  # good
    # print('logClustProb[vertexNr,:]', logClustProb[vertexNr, :])  # bad

    for k in range(nrClust):
      expDiffs = np.power(np.e, logClustProb - logClustProb[:, k][:, None])
      clustProb[:, k] = np.divide(1, np.sum(expDiffs, axis=1))

    for c in range(nrClust):
      print('sum%d' % c, np.sum(clustProb[:, c]))

    # import pdb
    # pdb.set_trace()

    return clustProb, crossData, longData


  def createLongData(self, data, diag, scanTimepts, partCode, ageAtScan):

    uniquePartCode = np.unique(partCode)

    longData = self.makeLongArray(data, scanTimepts, partCode, uniquePartCode)
    longDiagAllTmpts = self.makeLongArray(diag, scanTimepts, partCode, uniquePartCode)
    longDiag = np.array([x[0] for x in longDiagAllTmpts])
    longScanTimepts = self.makeLongArray(scanTimepts, scanTimepts, partCode, uniquePartCode)
    longPartCodeAllTimepts = self.makeLongArray(partCode, scanTimepts, partCode, uniquePartCode)
    longPartCode = np.array([x[0] for x in longPartCodeAllTimepts])
    longAgeAtScan = self.makeLongArray(ageAtScan, scanTimepts, partCode, uniquePartCode)
    uniquePartCodeFiltIndices = np.in1d(partCode, np.array(longPartCode))

    # filter cross-sectional data, keep only subjects with at least 2 visits
    filtData = data[uniquePartCodeFiltIndices,:]
    filtDiag = diag[uniquePartCodeFiltIndices]
    filtScanTimetps = scanTimepts[uniquePartCodeFiltIndices]
    filtPartCode = partCode[uniquePartCodeFiltIndices]
    filtAgeAtScan = ageAtScan[uniquePartCodeFiltIndices]
    inverseMap = np.squeeze(np.array([np.where(longPartCode == p) for p in filtPartCode])) # maps from longitudinal space
    #  to cross-sectional space

    print('partCode', partCode)
    print('filtPartCode', filtPartCode)
    print('longPartCode', longPartCode)
    print('inverseMap', inverseMap)
    print('np.max(inverseMap)', np.max(inverseMap))
    print('len(longData)', len(longData))
    assert(np.max(inverseMap) == len(longData)-1) # inverseMap indices should be smaller than the size of longData as they take elements from longData
    assert(len(inverseMap) == filtData.shape[0]) # length of inversemap should be the same as the cross-sectional data

    #print(np.max(inverseMap), len(longData), len(inverseMap), inverseMap.shape)
    #print(test)

    return longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan, inverseMap, \
      filtData, filtDiag, filtScanTimetps, filtPartCode, filtAgeAtScan, uniquePartCode

  def makeLongArray(self, array, scanTimepts, partCode, uniquePartCode):
    ''' place data in a longitudinal format, but only return a View
    (i.e. don't make a copy of the cross-sectional data). If the cross-data or long-data is modified,
    the changes will propagate in the other array. Useful for NaN inference. '''
    longArray = []  # longArray can be data, diag, ageAtScan,scanTimepts, etc .. both 1D or 2D
    nrParticipants = len(uniquePartCode)

    longCounter = 0

    for p in range(nrParticipants):
      # print('Participant %d' % uniquePartCode[p])
      currPartIndices = np.where(partCode == uniquePartCode[p])[0]
      currPartTimepoints = scanTimepts[currPartIndices]
      currPartTimeptsOrdInd = np.argsort(currPartTimepoints)
      # print uniquePartCode[p], currPartIndices, currPartTimepoints, currPartTimeptsOrdInd
      currPartIndicesOrd = currPartIndices[currPartTimeptsOrdInd]
      # print(uniquePartCode[p], currPartIndicesOrd)

      # for TADPOLE data we can even have x-sectional
      # assert (len(currPartTimeptsOrdInd) >= 2)  # 2 for PET, 3 for MRI

      # if len(currPartTimeptsOrdInd) > 1:
      longArray += [array[currPartIndicesOrd]]

      # print('array[currPartIndicesOrd[0],:]', array[currPartIndicesOrd[0], :])
      # print('longArray[-1]', longArray[-1][0])
      # array[currPartIndicesOrd[0], 0] = 2000
      # print('array[currPartIndicesOrd[0],:]', array[currPartIndicesOrd[0], :])
      # print('longArray[-1]', longArray[-1][0])
      # print(adsa)

    return longArray

  def postFitAnalysis(self, runPart, crossData, crossDiag, dpsCross, clustProbBCColNorm, paramsDataFile, resStruct):

    if runPart[4] == 'R':

      nrBiomk, nrClust = self.clustProb.shape
      plotTrajParams = self.params['plotTrajParams']
      nrSubjLong = self.subShifts.shape[0]
      longData = resStruct['longData']
      longDiag = resStruct['longDiag']
      scanTimepts = resStruct['scanTimepts']
      crossPartCode = resStruct['crossPartCode']

      longDPS = self.makeLongArray(dpsCross, scanTimepts, crossPartCode, np.unique(crossPartCode))

      # print('self.clustProb', self.clustProb)
      maxClustIndB = np.argmax(self.clustProb,axis=1)
      print('biomk groups:', [self.params['labels'][maxClustIndB == c] for c in range(nrClust)])
      print('self.thetas', self.thetas)
      print('self.variances', self.variances)
      print('self.subShifts', self.subShifts)


      thetasBiomkIndep = np.zeros((nrBiomk, 4))
      variancesBiomkIndep = np.zeros(nrBiomk)

      clustProbBiomkIndepCurr = np.identity(nrBiomk)

      # for b in range(nrBiomk):
      #   (thetasBiomkIndep[b, :], variancesBiomkIndep[b]) = self.estimThetas(crossData,
      #     dpsCross, clustProbBiomkIndepCurr[:, b], self.thetas[maxClustIndB[b]], nrSubjLong)

      thetasBiomkClust = [0 for x in range(nrClust)]
      for c in range(nrClust):
        thetasBiomkClust[c] = thetasBiomkIndep[maxClustIndB == c,:]

      # print('longDPS', longDPS[0].shape)
      # print('longData', longData[0])
      #
      # print('thetasBiomkIndep\n', thetasBiomkIndep)
      # print('self.thetas\n', self.thetas)

      # print(ads)

      # crossDataNaNs = crossData.copy()
      # crossDataNaNs[self.nanMask] = np.nan
      # crossDataNaNs

      # plot each biomarker line with the corresponding data

      # fig = self.plotterObj.plotTrajIndivBiomk(crossData, crossDiag, self.params['labels'], dpsCross,
      # longData, longDiag, longDPS,thetasBiomkIndep, variancesBiomkIndep, plotTrajParams, self.trajFunc,
      # nanMask=self.nanMask, replaceFigMode=True, showConfInt=False, colorTitle=False, yLimUseData=False,
      # adjustBottomHeight=0.25)
      # fig.savefig('%s/trajBiomkIndep.png' % self.outFolder, dpi=100)

    # plot each cluster with the biomarker trajectories that best match this cluster
      fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCross, longData,
        longDiag, longDPS, self.thetas, self.variances, clustProbBCColNorm,
        plotTrajParams, self.trajFunc, replaceFigMode=True, thetasSamplesClust=thetasBiomkClust,
        showConfInt=False, colorTitle=True, yLimUseData=False, adjustBottomHeight=0.25, orderClust=False,
        showInferredData=False)
      fig.savefig('%s/trajClustWithBiomkNoInfer.png' % self.outFolder, dpi=100)

      # print(adsa)
