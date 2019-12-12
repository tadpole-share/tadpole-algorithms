import numpy as np
import math
import scipy
#import scipy.cluster.hierarchy
import scipy.spatial
import pickle
import sklearn.cluster
import scipy.stats
import pdb
from plotFunc import *
import gc
from matplotlib import pyplot as pl

import DisProgBuilder
import os
import sys
import PlotterVDPM
import aux


class VoxelDPMBuilder(DisProgBuilder.DPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isCluster):
    self.plotterObj = PlotterVDPM.PlotterVDPM()

  def setPlotter(self, plotterObj):
    self.plotterObj = plotterObj

  def generate(self, dataIndices, expName, params):
    return VoxelDPM(dataIndices, expName, params, self.plotterObj)

class VoxelDPM(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, plotterObj):

    self.thetas = np.nan
    self.variances = np.nan
    self.subShifts = np.nan
    self.clustProb = np.nan
    self.dpsLongFirstVisit = np.nan

    self.params = params
    self.dataIndices = dataIndices

    self.expName = expName
    self.outFolder = 'resfiles/%s' % expName
    self.params['plotTrajParams']['outFolder'] = self.outFolder

    assert(params['data'].shape[0] == params['diag'].shape[0]
           == params['partCode'].shape[0] == params['ageAtScan'].shape[0]
           == dataIndices.shape[0] == params['scanTimepts'].shape[0])

    # can be informative or uniform(zero)
    self.logPriorShiftFunc = params['logPriorShiftFunc']
    self.logPriorShiftFuncDeriv = params['logPriorShiftFuncDeriv']
    self.paramsPriorShift = params['paramsPriorShift']
    self.logPriorThetaFunc = params['logPriorThetaFunc']
    self.logPriorThetaFuncDeriv = params['logPriorThetaFuncDeriv']
    self.paramsPriorTheta = None # set it later when you initialise thetas.

    self.plotterObj = plotterObj
    self.params['pdbPause'] = False

  def runStd(self, runPart):
    return self.run(runPart)

  def run(self, runPart):
    # filter some diagnostic groups, i.e. PCA/tAD or even for cross-validation
    filtParams = aux.filterDDSPAIndices(self.params, self.dataIndices)
    filtData = filtParams['data']
    filtDiag = filtParams['diag']
    filtScanTimepts = filtParams['scanTimepts']
    filtPartCode = filtParams['partCode']
    filtAgeAtScan = filtParams['ageAtScan']

    # filtData = filtParams[self.dataIndices, :]
    # filtDiag = self.params['diag'][self.dataIndices]
    # filtScanTimepts = self.params['scanTimepts'][self.dataIndices]
    # filtPartCode = self.params['partCode'][self.dataIndices]
    # filtAgeAtScan = self.params['ageAtScan'][self.dataIndices]

    # subjects that only contain one timepoint will be removed
    (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
     uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode) = \
      self.createLongData(filtData, filtDiag, filtScanTimepts, filtPartCode, filtAgeAtScan)
    nrClust = self.params['nrClust']
    plotTrajParams = self.params['plotTrajParams']

    # perform some initial clustering with k-means or use a-priori defined ROIs (freesurfer)
    initClust = self.runInitClust(runPart, crossData, crossDiag)
    if initClust is None:
      return None

    (nrSubjCross, nrBiomk) = crossData.shape
    nrSubjLong = len(longData)

    print(len(uniquePartCodeInverse), np.max(uniquePartCodeInverse), nrSubjLong, nrSubjCross)
    assert (np.max(uniquePartCodeInverse) < nrSubjLong)

    nrOuterIter = self.params['nrOuterIter']
    nrInnerIter = self.params['nrInnerIter']

    longAge1array = [np.concatenate((x.reshape(-1, 1), np.ones(x.reshape(-1, 1).shape,
      dtype=float)), axis=1) for x in longAgeAtScan]
    # np array NR_SUBJ_CROSS-SECTIONALLY x 2 [ cross-sectional_age 1]
    crossAge1array = np.concatenate((crossAgeAtScan.reshape(-1, 1),
      np.ones(crossAgeAtScan.reshape(-1, 1).shape,dtype=float)), axis=1)

    ageFirstVisitLong1array = np.array([s[0, :] for s in longAge1array])
    assert (ageFirstVisitLong1array.shape[1] == 2)

    # initialise cluster probabilities
    prevClustProbBC = aux.makeClustProbFromArray(initClust)

    # initialise subject specific shifts and rates
    initShiftsLong = np.nan * np.ones((nrSubjLong, 2), float)
    initShiftsLong[:, 0] = 1  # alpha
    longInitShifts = self.makeLongArray(self.params['initShift'], scanTimepts,
                                        crossPartCode, np.unique(crossPartCode))  # beta
    initShiftsLong[:, 1] = np.array([s[0] for s in longInitShifts])

    #print('longInitShifts', longInitShifts)
    #print(self.params['initShift'])
    #print(initShiftsLong)
    assert np.isfinite(initShiftsLong).all()

    # estimate some initial thetas, which are used as starting point in the numerical optimisatrion algo
    initThetas, initVariances = self.initTrajParams(crossData, crossDiag, prevClustProbBC,
      crossAgeAtScan, initShiftsLong, uniquePartCodeInverse, crossAge1array, self.params['rangeFactor'])

    # infer missing values in data from initial parameters
    initSubShiftsCross = initShiftsLong[uniquePartCodeInverse, :]

    self.plotterObj.nanMask = np.isnan(crossData)
    self.plotterObj.longDataNaNs = longData
    # assert np.sum(np.isnan(self.plotterObj.longDataNaNs[0])) > 0
    crossData, longData = self.inferMissingData(crossData, longData, prevClustProbBC, initThetas, initSubShiftsCross,
      crossAge1array, self.trajFunc, scanTimepts, crossPartCode, uniquePartCode, self.plotterObj)


    return self.runWithParams(runPart, longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
     uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode, nrClust,
     plotTrajParams, nrSubjCross, nrBiomk, nrSubjLong, nrOuterIter, nrInnerIter, longAge1array, crossAge1array, ageFirstVisitLong1array,
     prevClustProbBC, initShiftsLong, initThetas, initVariances)

  def calcGlobalMinimumStats(self, runPart):
    import sys

    print('step 0------------------')
    sys.stdout.flush()

    # print(adsa)
    # filter some diagnostic groups, i.e. PCA/tAD or even for cross-validation
    filtParams = aux.filterDDSPAIndices(self.params, self.dataIndices)
    filtData = filtParams['data']
    filtDiag = filtParams['diag']
    filtScanTimepts = filtParams['scanTimepts']
    filtPartCode = filtParams['partCode']
    filtAgeAtScan = filtParams['ageAtScan']

    # filtData = filtParams[self.dataIndices, :]
    # filtDiag = self.params['diag'][self.dataIndices]
    # filtScanTimepts = self.params['scanTimepts'][self.dataIndices]
    # filtPartCode = self.params['partCode'][self.dataIndices]
    # filtAgeAtScan = self.params['ageAtScan'][self.dataIndices]

    # subjects that only contain one timepoint will be removed
    print('Filt Part Code', filtPartCode)
    print('Filt Age At Scan', filtAgeAtScan)
    (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
     uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode) = \
      self.createLongData(filtData, filtDiag, filtScanTimepts, filtPartCode, filtAgeAtScan)
    nrClust = self.params['nrClust']
    plotTrajParams = self.params['plotTrajParams']

    print('step 1-----------------')
    sys.stdout.flush()

    # perform some initial clustering with k-means or use a-priori defined ROIs (freesurfer)
    initClust = self.runInitClust(runPart, crossData, crossDiag)
    if initClust is None:
      return None

    print('step 2---------')
    sys.stdout.flush()

    (nrSubjCross, nrBiomk) = crossData.shape
    nrSubjLong = len(longData)

    print(len(uniquePartCodeInverse), np.max(uniquePartCodeInverse), nrSubjLong, nrSubjCross)
    assert (np.max(uniquePartCodeInverse) < nrSubjLong)

    nrOuterIter = self.params['nrOuterIter']
    nrInnerIter = self.params['nrInnerIter']

    print('step 3------------------')

    longAge1array = [np.concatenate((x.reshape(-1, 1), np.ones(x.reshape(-1, 1).shape,
      dtype=float)), axis=1) for x in longAgeAtScan]
    # np array NR_SUBJ_CROSS-SECTIONALLY x 2 [ cross-sectional_age 1]
    crossAge1array = np.concatenate((crossAgeAtScan.reshape(-1, 1),
      np.ones(crossAgeAtScan.reshape(-1, 1).shape,dtype=float)), axis=1)

    ageFirstVisitLong1array = np.array([s[0, :] for s in longAge1array])
    assert (ageFirstVisitLong1array.shape[1] == 2)

    # initialise cluster probabilities
    prevClustProbBC = aux.makeClustProbFromArray(initClust)

    # initialise subject specific shifts and rates
    initShiftsLong = np.nan * np.ones((nrSubjLong, 2), float)
    initShiftsLong[:, 0] = 1  # alpha
    longInitShifts = self.makeLongArray(self.params['initShift'], scanTimepts,
                                        crossPartCode, np.unique(crossPartCode))  # beta
    initShiftsLong[:, 1] = np.array([s[0] for s in longInitShifts])

    print('longInitShifts', longInitShifts)
    print(self.params['initShift'])
    print(initShiftsLong)
    assert np.isfinite(initShiftsLong).all()

    print('step 4------------------')
    sys.stdout.flush()

    # estimate some initial thetas, which are used as starting point in the numerical optimisatrion algo
    initThetas, initVariances = self.initTrajParams(crossData, crossDiag, prevClustProbBC,
      crossAgeAtScan, initShiftsLong, uniquePartCodeInverse, crossAge1array, self.params['rangeFactor'])

    # infer missing values in data from initial parameters
    initSubShiftsCross = initShiftsLong[uniquePartCodeInverse, :]

    print('step 4.4------------------')
    sys.stdout.flush()

    self.plotterObj.nanMask = np.isnan(crossData)
    self.plotterObj.longDataNaNs = longData
    # assert np.sum(np.isnan(self.plotterObj.longDataNaNs[0])) > 0
    crossData, longData = self.inferMissingData(crossData, longData, prevClustProbBC, initThetas, initSubShiftsCross,
      crossAge1array, self.trajFunc, scanTimepts, crossPartCode, uniquePartCode, self.plotterObj)

    nrStartPoints = 20

    print('step 4.5------------------')
    sys.stdout.flush()

    runIndex = self.params['runIndex']

    if runIndex > 0:
      runPartGlobalMin = 'RRRII' # for running the experiment
      nrProcesses = self.params['nrProcesses']
      # from evaluationFramework import allocateRunIndicesToProcess
      # runInstances = allocateRunIndicesToProcess(nrStartPoints, nrProcesses, runIndex)
      runInstances = range(nrStartPoints)
    else:
      runInstances = range(nrStartPoints)
      runPartGlobalMin = 'LLLII'  # for loading saved results

    liks = np.zeros(nrStartPoints)
    resStructs = [[] for p in runInstances]
    expNameOrig = self.expName

    print('step 5------------------')
    sys.stdout.flush()

    resStructInformativeStart = self.runWithParams('LLLII', longData, longDiagAllTmpts, longDiag, longScanTimepts,
                                       longPartCode, longAgeAtScan,
                                       uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode,
                                       crossAgeAtScan, uniquePartCode, nrClust,
                                       plotTrajParams, nrSubjCross, nrBiomk, nrSubjLong, nrOuterIter, nrInnerIter,
                                       longAge1array, crossAge1array, ageFirstVisitLong1array,
                                       prevClustProbBC, initShiftsLong, initThetas, initVariances)

    for p in range(nrStartPoints):
      np.random.seed(p)

      initThetasPerturb = copy.deepcopy(initThetas)
      # get perturbed theta
      for par in range(initThetas.shape[1]):
        stdThetas = np.std(initThetas[:, par], axis=0)
        initThetasPerturb[:, par] = initThetas[:, par] + np.random.normal(
          np.zeros(initThetas[:, par].shape), stdThetas * np.ones(initThetas[:, par].shape))

      # get perturbed subject shifts
      initShiftsLongPerturb = copy.deepcopy(initShiftsLong)
      initShiftsLongPerturb[:, 0] = initShiftsLong[:, 0] + np.random.uniform(0.3, 3, initShiftsLong.shape[0])  # alpha
      initShiftsLongPerturb[:, 1] = initShiftsLong[:, 1] + np.random.normal(0, 1, initShiftsLong.shape[0])  # beta

      vertIndToPerturb = np.array(range(0, nrBiomk, 10))
      perm = np.array(range(nrBiomk))
      perm[vertIndToPerturb] = (nrBiomk-2) - perm[vertIndToPerturb]
      print('np.min(perm)', np.min(perm))
      print('np.max(perm)', np.max(perm))
      print('nrBiomk', nrBiomk)
      print('perm', perm[:30], perm[-30:])
      assert np.min(perm) == 0
      assert np.max(perm) == (nrBiomk - 1)

      prevClustProbPerturbBC = prevClustProbBC[perm, :]

      self.expName = '%s_pert%d' % (expNameOrig, p)
      self.outFolder = 'resfiles/%s' % self.expName
      self.params['plotTrajParams']['outFolder'] = self.outFolder
      os.system('mkdir -p %s' % self.outFolder)

      nrOuterIter = 5
      nrInnerIter = 1

      resStructs[p] = self.runWithParams(runPartGlobalMin, longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
       uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode, nrClust,
       plotTrajParams, nrSubjCross, nrBiomk, nrSubjLong, nrOuterIter, nrInnerIter, longAge1array, crossAge1array, ageFirstVisitLong1array,
       prevClustProbPerturbBC, initShiftsLongPerturb, initThetasPerturb, initVariances)

      liks[p] = resStructs[p]['lik']

      print('step 6------------------')
      sys.stdout.flush()

    infLik = resStructInformativeStart['lik']

    minLik = np.min(liks)
    minInd = np.argmin(liks)


    currCorrectAssign = np.zeros(nrStartPoints)
    diceAllGlobalMin = np.zeros(nrStartPoints)
    diceAllInf = np.zeros(nrStartPoints)

    for p in range(nrStartPoints):
      clustProbGlobalMin = resStructs[minInd]['clustProb']
      currCorrectAssign[p], currPerm, diceAllGlobalMin[p] = calcSpatialOverlap(clustProbGlobalMin, resStructs[p]['clustProb'])
      _, _, diceAllInf[p] = calcSpatialOverlap(resStructInformativeStart['clustProb'], resStructs[p]['clustProb'])


      print('%d\%', 100 * float(p+1) / nrStartPoints)

    print('step 7------------------')
    sys.stdout.flush()

    print('currCorrectAssign', currCorrectAssign)
    print('diceAllGlobalMin', diceAllGlobalMin)
    print('diceAllInf', diceAllInf)

    print('liks', liks)
    print('minLik', minLik)
    print('infLik', infLik)
    percAtGlobalMin = float(np.sum(np.abs(diceAllGlobalMin - np.max(diceAllGlobalMin)) < 0.1))/nrStartPoints
    print('percAtGlobalMin', percAtGlobalMin)

    print(ads)


  def runWithParams(self, runPart, longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
     uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode, nrClust,
     plotTrajParams, nrSubjCross, nrBiomk, nrSubjLong, nrOuterIter, nrInnerIter, longAge1array, crossAge1array, ageFirstVisitLong1array,
     prevClustProbBC, initShiftsLong, initThetas, initVariances):

    subShiftsLong = np.nan * np.ones((nrOuterIter, nrInnerIter, nrSubjLong, 2), float)
    subShiftsLong[0, 0, :, :] = initShiftsLong

    paramsDataFile = '%s/params_o%d.npz' % (self.outFolder, nrOuterIter)

    # initThetas, initVariances, subShiftsLongInit, \
    #   prevClustProbBC, paramsDataFile = self.loadParamsFromFile(paramsDataFile,
    #   nrOuterIter, nrInnerIter)
    # subShiftsLong[0, 0, :, :] = subShiftsLongInit

    plot = True

    if runPart[1] == 'R':

      clustProbOBC = np.zeros((nrOuterIter, nrBiomk, nrClust), float)
      clustProbOBC[0, :, :] = prevClustProbBC

      assert not np.isnan(initThetas).any()

      thetas = np.nan * np.ones((nrOuterIter, nrInnerIter, nrClust, initThetas.shape[1]), float)
      variances = np.nan * np.ones((nrOuterIter, nrInnerIter, nrClust), float)
      modelSpecificParams = np.nan * np.ones(nrOuterIter, float)

      thetas[0,0,:,:] = initThetas
      variances[0,0,:] = initVariances

      counter = 0
      prevOuterIt = 0
      prevInnerIt = 0

      prevSubShifts = subShiftsLong[prevOuterIt, prevInnerIt, :, :]
      prevThetas = thetas[prevOuterIt, prevInnerIt, :, :]
      prevVariances = variances[prevOuterIt, prevInnerIt, :]
      prevSubShiftsCross = prevSubShifts[uniquePartCodeInverse, :]
      dpsCross = VoxelDPM.calcDps(prevSubShiftsCross, crossAge1array)


      # print('crossData.shape', crossData.shape)
      # print(ads)

      for outerIt in range(nrOuterIter):

        # fit DPM
        prevClustProbBC = clustProbOBC[prevOuterIt, :,:]
        prevClustProbBCColNorm = prevClustProbBC/np.sum(prevClustProbBC,0)[None, :]

        # hueColsAtrophyExtentC, rgbColsAtrophyExtentC, sortedIndAtrophyExtent = \
        #   self.plotterObj.colorClustByAtrophyExtent(prevThetas, dpsCross, crossDiag,
        #     self.params['patientID'], self.trajFunc)

        # if plot:
        #   self.plotterObj.plotBlenderDirectGivenClustCols(hueColsAtrophyExtentC, prevClustProbBC, plotTrajParams,
        #     filePathNoExt='%s/atrophyExtent%d_%s' % (self.outFolder, outerIt, '_'.join(self.expName.split('/'))))

        print('outerIt ', outerIt)

        if outerIt == 2:
          self.params['pdbPause'] = True

        for innerIt in range(nrInnerIter):

          print('innerIt ', innerIt)
          sys.stdout.flush()

          prevSubShifts = subShiftsLong[prevOuterIt, prevInnerIt, :,:]
          prevThetas = thetas[prevOuterIt, prevInnerIt, :, :]
          prevVariances = variances[prevOuterIt, prevInnerIt, :]
          prevSubShiftsCross = prevSubShifts[uniquePartCodeInverse,:]

          dpsCross = VoxelDPM.calcDps(prevSubShiftsCross, crossAge1array)
          dpsLong = self.makeLongArray(dpsCross, scanTimepts, crossPartCode, np.unique(crossPartCode))



          # print('longAge1array', longAge1array)

          if plot:
            fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCross, longData, longDiag, dpsLong,
              prevThetas, prevVariances,
             prevClustProbBCColNorm, plotTrajParams, self.trajFunc, orderClust=True)
            fig.savefig('%s/loopMean%d%d1.png' % (self.outFolder, outerIt, innerIt), dpi = 100)

          # fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCross, longData, longDiag, dpsLong,
          #   prevThetas, prevVariances,
          #  prevClustProbBCColNorm, plotTrajParams, self.trajFunc, orderClust=True,
          #   showInferredData=True)
          # fig.savefig('%s/inferLoopMean%d%d1.png' % (self.outFolder, outerIt, innerIt), dpi = 100)


          assert np.isfinite(crossData).all()
          assert np.isfinite(crossDiag).all()
          assert np.isfinite(dpsCross).all()
          assert np.isfinite(prevThetas).all()
          assert np.isfinite(prevVariances).all()
          assert all([np.isfinite(a).all() for a in longAge1array])
          assert all([np.isfinite(a).all() for a in longData])
          assert np.isfinite(prevClustProbBC).all()
          assert np.isfinite(prevSubShifts).all()

          print('Estimate subject shifts')
          # estimate alphas and betas
          prevSubShiftAvg = np.mean(prevSubShifts, axis=0)

          for s in range(nrSubjLong):
            #print('Subject shift estimation %d/%d' % (s, nrSubjLong))
            subShiftsLong[outerIt, innerIt, s, :] = self.estimShifts(longData[s], prevThetas,
              prevVariances, longAge1array[s], prevClustProbBC, prevSubShifts[s,:], prevSubShiftAvg,
              self.params['fixSpeed'])

          sys.stdout.flush()
          subShiftsNewCross = subShiftsLong[outerIt, innerIt, uniquePartCodeInverse, :]
          dpsCrossNew = VoxelDPM.calcDps(subShiftsNewCross, crossAge1array)
          dpsLongNew = self.makeLongArray(dpsCrossNew, scanTimepts, crossPartCode, np.unique(crossPartCode))

          if plot:
            fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCrossNew, longData, longDiag,
              dpsLongNew, prevThetas, prevVariances, prevClustProbBCColNorm,
              plotTrajParams, self.trajFunc, orderClust=True)
            fig.savefig('%s/loopMean%d%d2.png' % (self.outFolder, outerIt, innerIt), dpi = 100)

          # fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCrossNew, longData, longDiag,
          #   dpsLongNew, prevThetas, prevVariances, prevClustProbBCColNorm,
          #   plotTrajParams, self.trajFunc, orderClust=True, showInferredData=True)
          # fig.savefig('%s/inferLoopMean%d%d2.png' % (self.outFolder, outerIt, innerIt), dpi = 100)

          # import pdb
          # pdb.set_trace()

          print('Estimate thetas and variances')
          # estimate thetas and variances
          for c in range(nrClust):
            print('Estimating theta %d/%d' % (c, nrClust))
            (thetas[outerIt, innerIt, c, :], variances[outerIt, innerIt, c]) = self.estimThetas(crossData,
              dpsCrossNew, prevClustProbBCColNorm[:,c], prevThetas[c,:], nrSubjLong, prevThetas)

          if plot:
            fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCrossNew,
              longData, longDiag, dpsLongNew, thetas[outerIt, innerIt, :, :],
              variances[outerIt, innerIt, :], prevClustProbBCColNorm, plotTrajParams, self.trajFunc,
              orderClust=True)
            fig.savefig('%s/loopMean%d%d3.png' % (self.outFolder, outerIt, innerIt), dpi = 100)

          # fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCrossNew,
          #   longData, longDiag, dpsLongNew, thetas[outerIt, innerIt, :, :],
          #   variances[outerIt, innerIt, :], prevClustProbBCColNorm, plotTrajParams, self.trajFunc,
          #   orderClust=True,  showInferredData=True)
          # fig.savefig('%s/inferLoopMean%d%d3.png' % (self.outFolder, outerIt, innerIt), dpi = 100)

          # no need to make shifts identifiable anymore, because we set informative priors on shifts,
          # which automatically makes them identifiable.
          # ensure alpha is positive and make the thetas identifiable
          # subShiftsLong[outerIt, innerIt, :, :], shiftTransform = VoxelDPM.makeShiftsIdentif(
          #   subShiftsLong[outerIt, innerIt,:,:], ageFirstVisitLong1array, longDiag)
          shiftTransform = [0,1] # mu_ctl, sigma_ctl - set them to not change anything

          # apart from making them identifiable also rescale them according to the DPS transformation
          thetas[outerIt, innerIt, :, :] = self.makeThetasIdentif(thetas[outerIt, innerIt,:,:], shiftTransform)

          counter += 1

          prevInnerIt = innerIt
          prevOuterIt = outerIt # need to also set outerIt otherwise at (2,1) it will use (1,0) instead of (2,0)

          # end of inner loop

        print('Recompute cluster assignments')
        # recopute responsibilities p(z_t = k) that voxel t was generated by cluster k
        subShiftsNewCross2 = subShiftsLong[outerIt, -1, uniquePartCodeInverse, :]
        clustProbOBC[outerIt, :, :], crossData, longData, modelSpecificParams[outerIt] = \
          self.recompResponsib(crossData, longData, crossAge1array,
          thetas[outerIt, -1, :, :], variances[outerIt, -1, :], subShiftsNewCross2, self.trajFunc,
          prevClustProbBC, scanTimepts, crossPartCode, uniquePartCode, outerIt)

      clustProbBC = clustProbOBC[-1,:,:]

      clustProbOBC = None
      gc.collect()
      print('freed up clustProbOBC')
      sys.stdout.flush()

      paramsStruct = dict(clustProbBC=clustProbBC,
        thetas=thetas, variances=variances,
        subShiftsLong=subShiftsLong, plotTrajParams=self.params['plotTrajParams'], modelSpecificParams=modelSpecificParams)

      pickle.dump(paramsStruct, open(paramsDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    elif runPart[1] == 'L':
        dataStruct = pickle.load(open(paramsDataFile, 'rb'))
        # dataStruct['plotTrajParams'] = self.params['plotTrajParams']
        # pickle.dump(dataStruct, open(paramsDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        # print(asdsa)

        clustProbBC = dataStruct['clustProbBC']
        thetas = dataStruct['thetas']
        variances = dataStruct['variances']
        subShiftsLong = dataStruct['subShiftsLong']
        modelSpecificParams = dataStruct['modelSpecificParams']
    elif runPart[1] == 'Non-enforcing':
      if os.path.isfile(paramsDataFile):
        dataStruct = pickle.load(open(paramsDataFile, 'rb'))
        if 'clustProbBC' in dataStruct.keys():
          clustProbBC = dataStruct['clustProbBC']
        else:
          print('dataStruct.keys()', dataStruct.keys())
          clustProbBC = dataStruct['clustProbOBC'][-1,:,:]

        print('clustProbBC.shape', clustProbBC.shape)
        #print('dataStruct[clustProbOBC].shape', dataStruct['clustProbOBC'].shape)
        thetas = dataStruct['thetas']
        variances = dataStruct['variances']
        subShiftsLong = dataStruct['subShiftsLong']
        modelSpecificParams = dataStruct['modelSpecificParams']
      else:
        print('no file found at runPart[2] for %s' % self.outFolder)
        return None
    else:
      raise ValueError('runpart needs to be either R, L or Non-enforcing')

    self.thetas = thetas[nrOuterIter - 1, nrInnerIter - 1, :, :]
    self.variances = variances[nrOuterIter - 1, nrInnerIter - 1, :]
    self.subShifts = subShiftsLong[nrOuterIter - 1, nrInnerIter - 1, :, :]
    self.clustProb = clustProbBC
    assert len(self.clustProb.shape) == 2
    clustProbBCColNorm = self.clustProb / np.sum(self.clustProb, 0)[None, :]
    subShiftsCross = self.subShifts[uniquePartCodeInverse, :]
    dpsCross = VoxelDPM.calcDps(subShiftsCross, crossAge1array)
    # import pdb
    # pdb.set_trace()
    self.dpsLongFirstVisit = VoxelDPM.calcDps(self.subShifts, ageFirstVisitLong1array)

    self.setModelSpecificParams(modelSpecificParams[-1])

    # self.clustProb, _, _, _ = self.recompResponsib(crossData, longData, crossAge1array,
    #   self.thetas, self.variances, subShiftsCross, self.trajFunc,
    #   self.clustProb, scanTimepts, crossPartCode, uniquePartCode, nrOuterIter)

    print('subShiftsLong.shape', subShiftsLong.shape)

    aic = np.nan
    bic = np.nan
    lik = np.nan
    bicFile = '%s/fitMetrics.npz' % self.outFolder
    if runPart[2] == 'R':
      lik, bic, aic = self.calcModelLogLikFromEnergy(crossData, dpsCross, self.thetas, self.variances,
                                               self.clustProb, nrSubjLong)
      dataStruct = dict(lik=lik, bic = bic, aic = aic)
      pickle.dump(dataStruct, open(bicFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    elif runPart[2] == 'L':
      dataStruct = pickle.load(open(bicFile, 'rb'))
      aic = dataStruct['aic']
      bic = dataStruct['bic']
      lik = dataStruct['lik']

    elif runPart[2] == 'Non-enforcing':
      if os.path.isfile(bicFile):
        dataStruct = pickle.load(open(bicFile, 'rb'))
        aic = dataStruct['aic']
        bic = dataStruct['bic']
        lik = dataStruct['lik']

    # lik2 = self.calcModelLogLik(crossData, dpsCross, self.thetas, self.variances, self.clustProb)


    # import pdb
    # pdb.set_trace()

    resStruct = dict(longData=longData, longDiagAllTmpts=longDiagAllTmpts, longDiag=longDiag,
      longScanTimepts=longScanTimepts, longPartCode=longPartCode, longAgeAtScan=longAgeAtScan,
      uniquePartCodeInverse=uniquePartCodeInverse, crossData=crossData, crossDiag=crossDiag,
      scanTimepts=scanTimepts, crossPartCode=crossPartCode, crossAgeAtScan=crossAgeAtScan,
      longAge1array=longAge1array, crossAge1array=crossAge1array, thetas=self.thetas,
      variances=self.variances, subShifts=self.subShifts, clustProb=self.clustProb,
      clustProbBCColNorm=clustProbBCColNorm, subShiftsCross=subShiftsCross,dpsCross=dpsCross,
      ageFirstVisitLong1array=ageFirstVisitLong1array, lik=lik, bic=bic, aic=aic,
      uniquePartCode=uniquePartCode, outFolder=self.outFolder)

    # otherVariables = dict()

    self.postFitAnalysis(runPart, crossData, crossDiag, dpsCross, clustProbBCColNorm, paramsDataFile, resStruct)

    return resStruct


  def estimShiftsInPlace(self, longData, prevThetas, prevVariances, longAge1array, prevClustProbBC, prevSubShifts, prevSubShiftAvg,
                fixSpeed, subShiftsLong, outerIt, innerIt, s):

    subShiftsLong[outerIt, innerIt, s, :] = self.estimShifts(longData, prevThetas,
                prevVariances, longAge1array, prevClustProbBC, prevSubShifts, prevSubShiftAvg,
                fixSpeed)

  def calcPredScores(self, testInputInd, testPredInd, filteredParams):
    ''' Predict future biomarker values given two initial scans.
    Compare accuracy with actual measured values. '''


    partCodeTestInitial = filteredParams['partCode'][testInputInd]
    unqPartCode = np.unique(partCodeTestInitial)
    maskTwoMore = np.zeros(testInputInd.shape, bool)
    for p in range(unqPartCode.shape[0]):
      if np.sum(partCodeTestInitial == unqPartCode[p]) >= 2:
        maskTwoMore[filteredParams['partCode'] == unqPartCode[p]] = True

    print('maskTwoMore', maskTwoMore)
    print('filteredParams[partCode]', filteredParams['partCode'])
    print('testInputInd', np.sum(testInputInd))
    testInputInd = np.logical_and(testInputInd, maskTwoMore)
    print('testInputInd', np.sum(testInputInd))
    print('testPredInd', np.sum(testPredInd))
    # print(adsa)

    # set fixed speed for predicting biomarker values, otherwise 1-2 subjects will get high, negative alphas
    # due to noise in measurements
    # UPDATE: now using informative priors on speed and time shift, no need for fixing speed.
    # self.params['fixSpeed'] = True
    maxLikStages, _, _, _, _, otherParams = self.stageSubjects(testInputInd)

    # maxLikStagesFS, _, _, _, _, otherParamsFS = self.stageSubjects(testInputInd, fixSpeed=True)

    subShiftsLong = otherParams['subShiftsLong']
    subShiftsCross = otherParams['subShiftsCross']
    testInputPartCode = otherParams['longPartCode']
    testInputDPS = self.calcDpsNo1array(subShiftsCross,
      filteredParams['ageAtScan'][testInputInd])

    print('maxLikStages', maxLikStages)
    print('subShiftsLong', subShiftsLong)
    # print('maxLikStagesFS', maxLikStagesFS)
    # print('subShiftsLongFS', otherParamsFS['subShiftsLong'])
    # import pdb
    # pdb.set_trace()

    testPredCrossPartCode = filteredParams['partCode'][testPredInd]
    testPredCrossAge = filteredParams['ageAtScan'][testPredInd]
    testPredSubShifts = np.zeros((testPredCrossAge.shape[0], 2), float)

    for p in range(testInputPartCode.shape[0]):
      matchInd = np.in1d(testPredCrossPartCode, testInputPartCode[p])
      print('subShiftsLong.shape', subShiftsLong.shape)
      print('testPredSubShifts[matchInd,:].shape', testPredSubShifts[matchInd, :].shape)
      testPredSubShifts[matchInd, :] = subShiftsLong[p, :]

    print('testInputPartCode', testInputPartCode)
    print('testPredCrossPartCode', testPredCrossPartCode)
    print('testPredCrossAge', testPredCrossAge)
    print('testPredSubShifts', testPredSubShifts)

    testPredDPS = self.calcDpsNo1array(testPredSubShifts, testPredCrossAge)

    testPredData = filteredParams['data'][testPredInd]
    nrSubjPredCross = testPredDPS.shape[0]
    nrBiomk, nrClust = self.clustProb.shape
    testPredClust = np.zeros((nrSubjPredCross, nrClust), float)
    for c in range(nrClust):
      testPredClust[:, c] = self.trajFunc(testPredDPS, self.thetas[c, :])

    assert (np.abs(np.sum(self.clustProb, axis=1) - 1) < 0.001).all()

    print('testPredDPS', testPredDPS)
    print('testInputDPS', testInputDPS)

    # actual biomk predictions for each subject and each biomk
    testPredPredPB = np.dot(testPredClust, self.clustProb.T)
    dataPredDiff = (testPredPredPB - testPredData)
    RMSE = np.sqrt(np.mean((dataPredDiff ** 2).astype(np.longdouble), axis=(0, 1)))
    RMSEfromZero = np.sqrt(np.mean((testPredData ** 2).astype(np.longdouble), axis=(0, 1)))

    print('dataPredDiff[::10000,::10000]', dataPredDiff[::10000, ::10000])
    print('RMSE', RMSE)
    print('RMSEfromZero', RMSEfromZero)
    norm1Diff = np.sum(np.abs(dataPredDiff).astype(np.longdouble), axis=(0, 1))
    norm1DiffFromZero = np.sum(np.abs(testPredData).astype(np.longdouble), axis=(0, 1))
    print('norm1Diff', norm1Diff)
    print('norm1DiffFromZero', norm1DiffFromZero)

    testData = np.concatenate((filteredParams['data'][testInputInd],
    filteredParams['data'][testPredInd]), axis=0)
    testDiag = np.concatenate((filteredParams['diag'][testInputInd],
    filteredParams['diag'][testPredInd]), axis=0)
    testDPSCross = np.concatenate((testInputDPS, testPredDPS))
    testScanTimepts = np.concatenate((filteredParams['scanTimepts'][testInputInd],
    filteredParams['scanTimepts'][testPredInd]), axis=0)
    testPartCode = np.concatenate((filteredParams['partCode'][testInputInd],
    filteredParams['partCode'][testPredInd]), axis=0)
    testAgeAtScan = np.concatenate((filteredParams['ageAtScan'][testInputInd],
    filteredParams['ageAtScan'][testPredInd]), axis=0)

    (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
    uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode) = \
      self.createLongData(testData, testDiag, testScanTimepts, testPartCode, testAgeAtScan)

    dpsLongNew = self.makeLongArray(testDPSCross, testScanTimepts, testPartCode,
      np.unique(testPartCode))

    print(testData.shape[0], testDiag.shape[0], testDPSCross.shape[0])
    assert testData.shape[0] == testDiag.shape[0] == testDPSCross.shape[0]
    print(len(longData), longDiag.shape[0], len(dpsLongNew))
    assert len(longData) == longDiag.shape[0] == len(dpsLongNew)

    for p in range(len(longData)):
      print('%d' % p, 'dps', dpsLongNew[p], 'subShifts ', subShiftsLong[p, :])

    # pl.figure(3)
    clustProbBCColNorm = self.clustProb / np.sum(self.clustProb, 0)[None, :]
    fig = self.plotterObj.plotTrajWeightedDataMean(testData, testDiag, testDPSCross,
      longData, longDiag, dpsLongNew, self.thetas, self.variances,
      clustProbBCColNorm, self.params['plotTrajParams'], self.trajFunc, orderClust=True,
      replaceFigMode=True)
    fig.savefig('%s/predScoresTestDataTraj_%s.png' %
                (self.outFolder, '_'.join(self.expName.split('/'))), dpi=100)



    # plot the mean difference and individual differences
    # meanAbsDiffB = np.sum(np.abs(dataPredDiff), axis=0)
    # self.plotterObj.plotDiffs(meanAbsDiffB, self.params['plotTrajParams'],
    #   filePathNoExt='%s/diffsMean_%s' % (self.outFolder, '_'.join(self.expName.split('/'))))

    # nrSubjTestPred = dataPredDiff.shape[0]
    # for p in range(nrSubjTestPred):
    #   absDiffB = np.abs(dataPredDiff[p,:])
    #   self.plotterObj.plotDiffs(absDiffB, self.params['plotTrajParams'],
    #     filePathNoExt='%s/diffs_p%d_%s' % (self.outFolder, p, '_'.join(self.expName.split('/'))))

    # print(adsas)

    return RMSE, testPredPredPB, testPredData


  @staticmethod
  def makeShiftsIdentif(subShiftsLong, ageFirstVisitLong1array, longDiag):
    # make the subject shifts identifiable so that the dps_score(controls) ~ N(0,1) (as in Jedynak, 2012)
    dpsLong = VoxelDPM.calcDps(subShiftsLong, ageFirstVisitLong1array)
    dpsCTL = dpsLong[longDiag == CTL]
    muCTL = np.mean(dpsCTL)
    sigmaCTL = np.std(dpsCTL)
    # need to do: alphas = alphas / std_ctl    betas = (betas - mean_clt)/std_ctl
    subShiftsLong[:, 1] -= muCTL # betas = (betas - mean_clt)
    subShiftsLong[:, :] /= sigmaCTL # alphas = alphas / std_ctl    betas = betas/std_ctl

    shiftTransform = [muCTL, sigmaCTL]

    return subShiftsLong, shiftTransform

  @staticmethod
  def makeThetasIdentif(thetas, shiftTransform):
    """
    Make the parameters theta so that b > 0. Since most traj will be decreasing,
    we will have that a < 0 and d > 0. The minimum will be a, and maximum will be a+d

    shiftTransform = [muCTL, sigmaCTL] """

    nrClust = thetas.shape[0]
    for c in range(nrClust):
      if thetas[c, 1] < 0:
        # d_k = d_k + a_k
        thetas[c, 3] = thetas[c, 3] + thetas[c, 0]
        # a_k = -a_k  b_k = -b_k
        thetas[c, [0, 1]] = -thetas[c, [0, 1]]

    #re-transform the parameters using the DPS transformation

    # thetas[:,1] /= shiftTransform[1] # b = b / sigma
    # thetas[:,2] = shiftTransform[0] + shiftTransform[1] * thetas[:,2] # c = mu + sigma*c

    thetas[:, 1] *= shiftTransform[1]  # b = b * sigma
    thetas[:, 2] = (thetas[:, 2] - shiftTransform[0])/shiftTransform[1]   # c = (c - mu) / sigma

    return thetas

  @staticmethod
  def normDPS(subShiftsLong, ageFirstVisitLong1array, longDiag, uniquePartCodeInverse, crossAge1array,
    thetas, thetasSamples):

    subShiftsLongNorm = copy.deepcopy(subShiftsLong)
    # normalise the DPS so that DPS of controls ~ N(0,1)
    subShiftsLongNorm, shiftTransform = VoxelDPM.makeShiftsIdentif(
      subShiftsLongNorm, ageFirstVisitLong1array, longDiag)
    subShiftsCrossNorm = subShiftsLongNorm[uniquePartCodeInverse, :]

    dpsLongNorm = VoxelDPM.calcDps(subShiftsLongNorm, ageFirstVisitLong1array)
    dpsLong = VoxelDPM.calcDps(subShiftsLong, ageFirstVisitLong1array)
    dpsCrossNorm = VoxelDPM.calcDps(subShiftsCrossNorm, crossAge1array)

    # print('subShiftsLong', subShiftsLong)
    # print('subShiftsLongNorm', subShiftsLongNorm)
    # print('dpsLong', dpsLong)
    # print('dpsLongNorm', dpsLongNorm)
    # print('shiftTransform', shiftTransform)

    # apart from making them identifiable also rescale them according to the DPS transformation
    thetasNorm = VoxelDPM.makeThetasIdentif(thetas, shiftTransform)

    if thetasSamples is not None:
      thetasSamplesNorm = copy.deepcopy(thetasSamples)

      for s in range(thetasSamplesNorm.shape[1]):
        thetasSamplesNorm[:,s,:] = VoxelDPM.makeThetasIdentif(thetasSamplesNorm[:,s,:], shiftTransform)
    else:
      thetasSamplesNorm = None

    return subShiftsLongNorm, subShiftsCrossNorm, dpsLongNorm, dpsCrossNorm, thetasNorm, thetasSamplesNorm

  def initTrajParams(self, crossData, crossDiag, clustProbBC, crossAgeAtScan, subShiftsLong,
                     uniquePartCodeInverse, crossAge1array, extraRangeFactor):
    assert not np.isnan(crossData).any()
    assert not np.isnan(crossDiag).any()
    assert not np.isnan(clustProbBC).any()
    assert not np.isnan(crossAgeAtScan).any()
    assert not np.isnan(subShiftsLong).any()
    assert not np.isnan(uniquePartCodeInverse).any()
    assert not np.isnan(crossAge1array).any()

    nrClust = clustProbBC.shape[1]
    initSubShiftsCross = subShiftsLong[uniquePartCodeInverse, :]
    variances = np.zeros(nrClust, float)
    dpsCross = VoxelDPM.calcDps(initSubShiftsCross, crossAge1array)

    # calculate average voxel value for each (subject, cluster) pair, use them to initialise theta
    clustProbColNorm = clustProbBC / np.sum(clustProbBC, 0)[None, :]  # p(z_t = c) / (\sum_{t=1}^T) p(z_t = c)
    avgVoxelSC = np.dot(crossData, clustProbColNorm)  # SUBJECTS x CLUSTER array of avg voxel values
    thetas = np.zeros((nrClust, 4), float)
    minMaxRange = np.max(avgVoxelSC, 0) - np.min(avgVoxelSC, 0)
    assert (1 <= self.params['patientID'] <= 3)

    # estimate biomk direction from data, i.e. compare mean voxel value for CTL vs AD
    # meanBiomkPAT = np.mean(crossData[crossDiag == self.params['patientID'],:],axis=(0,1))
    # meanBiomkCTL = np.mean(crossData[crossDiag == CTL, :], axis=(0, 1))
    # if meanBiomkCTL < meanBiomkPAT:
    #   estimSlopeSign = 1 # in case of amyloid PET
    # else:
    #   estimSlopeSign = -1 # in case of cortical thickness

    # print('crossDiag', crossDiag, 'patientID', self.params['patientID'])
    dpsCrossPat = np.std(dpsCross[crossDiag == self.params['patientID']])
    # print(crossAge1array)
    # print(initSubShiftsCross)
    # print(np.std(dpsCross))
    # print(ads)

    assert (dpsCrossPat != 0 and np.isfinite(dpsCrossPat))

    thetas[:, 3] = np.min(avgVoxelSC, 0) - minMaxRange * extraRangeFactor / 10  # dk
    thetas[:, 0] = minMaxRange + minMaxRange * 2 * extraRangeFactor / 10  # ak = (dk + ak) - dk
    # t-time = 4/slope = 16/(ak*bk) = np.std(dpsCross) => bk = 16/(ak*std(dpsCross))
    transitionTime = 4 * np.std(dpsCross) # mean +/- 2sigma, covers 95% of data under normality assumption
    thetas[:, 1] = -16 / (thetas[:, 0] * transitionTime)  # bk = 4/ak so that slope = bk*ak/4 = -1
    thetas[:, 2] = np.mean(crossAgeAtScan)  # ck

    thetas = self.makeThetasIdentif(thetas, shiftTransform=[0, 1])

    # print('thetas', thetas)
    # print(adas)

    if self.params['informPrior']:
      """ WARNING. consider sigmoid with b >0, which means that minimum is a < 0, maximum is a+d.

      set prior for sigmoidal params [a,b,c,d] with minimum a, maximum a+d, slope a*b/4
      and slope maximum attained at center c
      f(s|theta = [a,b,c,d]) = a/(1+exp(-b(s-c)))+d
      These priors are very weak, but data driven. While data-driven priors can be
      controversial, it is clearly better than fixing some parameters using a
      data-driven estimate.
      """
      avgTheta = np.mean(thetas,axis=0) # find an average sigmoid for all voxels
      mu_a = avgTheta[0]
      mu_b = avgTheta[1]
      mu_c = avgTheta[2]
      mu_d = avgTheta[3]
      std_a = np.mean(minMaxRange)/2 # more informative prior
      std_b = 100 # set large number to make non-informative
      std_c = 100 # set large number to make non-informative
      std_d = std_a # more informative prior

      self.paramsPriorTheta = [mu_a, std_a, mu_b, std_b, mu_c, std_c, mu_d, std_d]

    for c in range(nrClust):
      # set zero subj because it's only used for DOF correction of subject shifts, and so far no shifts have been estimated.
      variances[c] = 1/3*self.estimVariance(crossData, dpsCross,
        clustProbColNorm[:,c], thetas[c,:], nrSubjLong=0)

    # print('std dpsCrossPat', np.std(dpsCross[crossDiag == self.params['patientID']]))
    # print('thetas', thetas, variances, crossData.shape, clustProbBC.shape)
    #print(asdsa)

    return thetas, variances

  def calcModelLogLik(self, data, dpsCross, thetas, variances, _):
    """ computed the full model log likelihood, used for checking if it increases during EM and for BIC"""

    nrBiomk = data.shape[1]
    nrClust = thetas.shape[0]

    prodLikBC = np.zeros((nrBiomk, nrClust), float)

    for c in range(nrClust):

      sqErrorsSB = np.power((data - self.trajFunc(dpsCross, thetas[c,:])[:, None]), 2) # taken from estimTheta
      pdfSB = (2*math.pi*variances[c])**(-1) * np.exp(-(2*variances[c])**(-1) * sqErrorsSB)

      #np.prod(pdfSB, axis=0)
      prodLikBC[:,c] = (1/nrClust) * np.prod(pdfSB, axis=0) # it is product here as the log doesn't go this far

      # prodLikBC[b,c] = (1/nrClust) * np.prod(scipy.stats.norm(data[:,b], loc=self.trajFunc(dpsCross, thetas[c,:]),
      #   scale=variances[c]))

    logLik = np.sum(np.log(np.sum(prodLikBC, axis=1)))

    print('logLik', logLik)
    import pdb
    pdb.set_trace()

    return logLik

  def calcModelLogLikFromEnergy(self, data, dpsCross, thetas, variances, clustProbBC, nrSubjLong):
    """ computed the full model log likelihood from the Energy (used in EM) and Entropy over Z"""

    nrBiomk = data.shape[1]
    nrClust = thetas.shape[0]

    # first calculate the energy term used in EM
    prodLogLikBC = np.zeros((nrBiomk, nrClust), float)
    for c in range(nrClust):

      sqErrorsSB = np.power((data - self.trajFunc(dpsCross, thetas[c,:])[:, None]), 2) # taken from estimTheta
      logpdfSB = -np.log(2*math.pi*variances[c])/2 - (2*variances[c])**(-1) * sqErrorsSB

      #np.prod(pdfSB, axis=0)
      prodLogLikBC[:,c] = np.sum(clustProbBC[:,c][None,:] * np.sum(logpdfSB, axis=0), axis=0)

      # prodLikBC[b,c] = (1/nrClust) * np.prod(scipy.stats.norm(data[:,b], loc=self.trajFunc(dpsCross, thetas[c,:]),
      #   scale=variances[c]))

    logLikEnergy = np.sum(prodLogLikBC, axis=(0,1))

    # calculate the entropy term
    logClustProbBC = np.nan_to_num(np.log(clustProbBC))
    # logClustProbBC[np.isnan(logClustProbBC)] = 0
    logLikEntropy = -np.sum(clustProbBC * logClustProbBC, axis=(0,1))

    logLik = logLikEnergy + logLikEntropy

    print('logLikEnergy', logLikEnergy)
    print('logLikEntropy', logLikEntropy)
    print('logLik', logLik)

    # import pdb
    # pdb.set_trace()

    # calculate BIC and AIC
    nrDataPoints = data.shape[0]*data.shape[1]

    # should I include the clustering probabilities in the nr Free params? No as they are marginalised in the model
    nrFreeParams = nrClust * (thetas.shape[1]+1) + 2*nrSubjLong
    # nrFreeParams = nrFreeParams + clustProbBC.shape[0] * clustProbBC.shape[1]

    bic = -2*logLik + nrFreeParams * np.log(nrDataPoints)
    aic = -2*logLik + nrFreeParams * 2

    print('bic')

    return logLik, bic, aic


  def estimThetas(self, data, dpsCross, clustProbColNormB, prevTheta, nrSubjLong, prevThetas):
    '''for sigmoidal trajectories, only fit slope and center, leave min and max fixed'''

    recompThetaSig = lambda thetaFull, theta12: [thetaFull[0],theta12[0],theta12[1],thetaFull[3]]

    objFunc = lambda theta12: self.objFunTheta(recompThetaSig(prevTheta, theta12),
                                               data, dpsCross, clustProbColNormB)[0]
    # objFuncDeriv = lambda theta12: self.objFunThetaDeriv(recompThetaSig(prevTheta, theta12),
    #                                                      data, dpsCross, clustProbColNormB)[[1, 2]]

    initTheta12 = prevTheta[[1,2]]
    # res = scipy.optimize.minimize(objFunc, initTheta12, method='BFGS', jac=objFuncDeriv,
    #                               options={'gtol': 1e-8, 'disp': True, 'maxiter':70})
    res = scipy.optimize.minimize(objFunc, initTheta12, method='Nelder-Mead',
                                  options={'xtol': 1e-8, 'disp': True, 'maxiter':70})


    # print(res)
    # print('-----------------------')
    # print(res2)
    # print(res.x, res2.x, initTheta12)
    # print(objFunc(res.x),objFunc(res2.x), objFunc(initTheta12))
    #
    # assert(np.abs(objFunc(res.x) - objFunc(res2.x)) < 0.001)

    newTheta = recompThetaSig(prevTheta, res.x)
    #print(newTheta)
    newVariance = self.estimVariance(data, dpsCross, clustProbColNormB, newTheta, nrSubjLong)

    return newTheta, newVariance

  def sampleThetas(self, data, dpsCross, clustProbB, initTheta, initVariance, nrSubjLong, nrSamples, propCovMat):

    objFunc = lambda params: self.objFunThetaLogL(params[:-1], params[-1], data, dpsCross, clustProbB)

    nrAccSamples = 0
    currSample = np.array(list(initTheta) + [initVariance])
    paramsSamples = np.zeros((nrSamples, initTheta.shape[0] + 1), float)
    paramsSamples[0] = currSample
    currLogL = objFunc(currSample)
    logL = np.zeros((nrSamples, 1), float)
    logL[0] = currLogL

    stds = np.sqrt(np.diag(propCovMat))+0.000001
    print('stds', stds)
    import time

    for s in range(1,nrSamples):
      print(s)

      newSample = np.random.multivariate_normal(currSample, propCovMat)
      # newSample = np.array([np.random.normal(currSample[i], stds[i]) for i in range(currSample.shape[0])])

      newLogL = objFunc(newSample)

      # print(np.log(2))
      # print(np.random.rand())
      # print(np.log(np.random.rand))

      if newLogL - currLogL > np.log(np.random.rand()):
        # accept sample
        currSample = newSample
        currLogL = newLogL
        nrAccSamples += 1

      paramsSamples[s, :] = currSample
      logL[s,:] = currLogL

      print('currSample', currSample, 'logL', logL[s])

    accRatio = nrAccSamples/nrSamples

    print('logL ', logL, 'accRatio  ', accRatio)

    return paramsSamples[:,:-1], paramsSamples[:,-1]

  def objFunThetaLogL(self, theta, variance, data, dpsCross, clustProbB):

    # print(theta)
    # print(variance)

    sqErrorsSB = -np.log(2*math.pi*variance)/2 - (1/(2*variance))*np.power((data - self.trajFunc(dpsCross, theta)[:, None]),2)
    meanLogL = np.sum(clustProbB[None,:] * sqErrorsSB, (0,1))

    return meanLogL

  def objFunTheta(self, theta, data, dpsCross, clustProbB):

    #print(subShiftsCross.shape, crossAge1array.shape, dps.shape, theta.shape, self.trajFunc(dps, theta).shape, data.shape)
    #print(test2)
    # currTheta = prevTheta
    # currTheta[1:2] = theta12
    # print('theta', theta)
    sqErrorsSB = np.power((data - self.trajFunc(dpsCross, theta)[:, None]),2)
    meanSSD = np.sum(np.multiply(clustProbB[None,:], sqErrorsSB), (0,1))
    #print("meanSSD", meanSSD, "clustProbB", clustProbB, "sqErrorsSB", sqErrorsSB)
    #print("ssd/nrSubj", meanSSD/data.shape[0])
    #print(asdsa)

    logPriorTheta = self.logPriorThetaFunc(theta, self.paramsPriorTheta)

    return meanSSD - logPriorTheta, meanSSD

  def objFunThetaDeriv(self, theta, dataSB, dpsCrossS, clustProbB):

    aK = theta[0]
    bK = theta[1]
    cK = theta[2]

    errorsSB = 2*(dataSB - self.trajFunc(dpsCrossS, theta)[:, None])

    # -ak (1+exp(-bk(alpha_i * t_ij + beta_i - ck)))^-2
    derivTerm1S = -aK * np.power(1 + np.exp(-bK*(dpsCrossS - cK)),-2)

    # *= exp(-bk(alpha_i * t_ij + beta_i - ck))
    derivTerm12S = derivTerm1S * np.exp(-bK * (dpsCrossS - cK))

    akDerivTermS = -np.power(1 + np.exp(-bK*(dpsCrossS - cK)),-1)
    bkDerivTermS = -derivTerm12S * -(dpsCrossS - cK)
    ckDerivTermS = -derivTerm12S * bK
    dkDerivTermS = -np.array(1)

    akSqErrorDerivSB = errorsSB * akDerivTermS[:,None]
    bkSqErrorDerivSB = errorsSB * bkDerivTermS[:,None]
    ckSqErrorDerivSB = errorsSB * ckDerivTermS[:,None]
    dkSqErrorDerivSB = errorsSB * dkDerivTermS

    akMeanSEDeriv = np.sum(clustProbB[None,:] * akSqErrorDerivSB, (0,1))
    bkMeanSEDeriv = np.sum(clustProbB[None, :] * bkSqErrorDerivSB, (0, 1))
    ckMeanSEDeriv = np.sum(clustProbB[None, :] * ckSqErrorDerivSB, (0, 1))
    dkMeanSEDeriv = np.sum(clustProbB[None, :] * dkSqErrorDerivSB, (0, 1))

    return np.array([akMeanSEDeriv, bkMeanSEDeriv, ckMeanSEDeriv, dkMeanSEDeriv])


  def estimShifts(self, dataOneSubj, thetas, variances, ageOneSubj1array, clustProb, prevSubShift,
    prevSubShiftAvg, fixSpeed):

    if fixSpeed:
      return self.estimShiftsBetaOnly(dataOneSubj, thetas, variances, ageOneSubj1array, clustProb,
        prevSubShift)

    objFunc = lambda shift: self.objFunShift(shift, dataOneSubj, thetas, variances, ageOneSubj1array, clustProb)
    # objFuncDeriv = lambda shift: self.objFunShiftDeriv(shift, dataOneSubj, thetas, variances, ageOneSubj1array,
    #                                                          clustProb)

    # res = scipy.optimize.minimize(objFunc, prevSubShift, method='BFGS', jac=objFuncDeriv,
    #                               options={'gtol': 1e-8, 'disp': True})

    res = scipy.optimize.minimize(objFunc, prevSubShift, method='Nelder-Mead',
                                  options={'xtol': 1e-8, 'disp': True, 'maxiter':70})

    newShift = res.x

    if newShift[0] < -400: # and -67
      import pdb
      pdb.set_trace()

    return newShift

  def estimShiftsBetaOnly(self, dataOneSubj, thetas, variances, ageOneSubj1array, clustProb, prevSubShift):

    alpha = prevSubShift[0]
    reconstructShifts = lambda beta: np.array([alpha, beta])

    objFunc = lambda beta: self.objFunShift(reconstructShifts(beta), dataOneSubj, thetas,
                                             variances, ageOneSubj1array, clustProb)

    initBeta = prevSubShift[1]

    print('ageOneSubj1array', ageOneSubj1array)
    print(objFunc(initBeta))
    import pdb
    pdb.set_trace()

    res = scipy.optimize.minimize(objFunc, initBeta, method='Nelder-Mead',
                                  options={'xtol': 1e-8, 'disp': True, 'maxiter':70})


    newShift = reconstructShifts(res.x)

    return newShift

  def objFunShift(self, shift, dataOneSubj, thetas, variances, ageOneSubj1array, clustProb):

    dps = np.sum(np.multiply(shift, ageOneSubj1array),1)
    nrClust = clustProb.shape[1]
    #for tp in range(dataOneSubj.shape[0]):
    sumSSD = 0
    for k in range(nrClust):
      sqErrorsB = np.sum(np.power((dataOneSubj - self.trajFunc(dps, thetas[k,:])[:, None]),2), axis=0)
      sumSSD += np.sum(np.multiply(sqErrorsB, clustProb[:,k]))/(2*variances[k])

    logPriorShift = self.logPriorShiftFunc(shift, self.paramsPriorShift)

    # print('logPriorShift', logPriorShift, 'sumSSD', sumSSD)
    # print(sumSSD)
    # if shift[0] < -400: # and -67
    #   import pdb
    #   pdb.set_trace()

    return sumSSD - logPriorShift

  def objFunShiftDeriv(self, shift, dataOneSubjTB, thetas, variances, ageOneSubj1array, clustProb):

    aK = thetas[:, 0]
    bK = thetas[:, 1]
    cK = thetas[:, 2]

    dpsT = np.sum(shift * ageOneSubj1array,1)
    #print(ageOneSubj1array.shape)
    nrClust = clustProb.shape[1]
    #for tp in range(dataOneSubj.shape[0]):
    sumSqErrorDerivalpha = 0
    sumSqErrorDerivbeta = 0
    for k in range(nrClust):
      errorsTB = dataOneSubjTB - self.trajFunc(dpsT, thetas[k, :])[:, None]

      # ak*(1+exp(-bk(alpha_i * t_ij + beta_i - ck)))^-2
      derivTerm1T = aK[k]*np.power((1+np.exp(-bK[k]*(dpsT-cK[k]))),-2)

      # exp(-bk(alpha_i * t_ij + beta_i - ck))
      derivTerm2T = derivTerm1T * np.exp(-bK[k]*(dpsT-cK[k]))

      # *= -bk * t_ij
      alphaDerivTermT = derivTerm2T * (-bK[k] * ageOneSubj1array[:,0])
      # *= -bk
      betaDerivTermT = derivTerm2T * -bK[k]

      alphaSqErrorDerivTB = (2 * errorsTB * alphaDerivTermT[:,None])
      betaSqErrorDerivTB = (2 * errorsTB * betaDerivTermT[:,None])

      sumSqErrorDerivalpha += np.sum(np.sum(alphaSqErrorDerivTB, 0) * clustProb[:,k])/ (2*variances[k])
      sumSqErrorDerivbeta += np.sum(np.sum(betaSqErrorDerivTB, 0) * clustProb[:, k]) / (2*variances[k])

    logPriorShiftDeriv = self.logPriorShiftFuncDeriv(shift, self.paramsPriorShift)


    if sumSqErrorDerivalpha == 0:
      pdb.set_trace()

    return np.array([sumSqErrorDerivalpha - logPriorShiftDeriv[0], sumSqErrorDerivbeta - logPriorShiftDeriv[1]])

  def setModelSpecificParams(self, modelSpecificParamsFinal):
    ''' only implement in submodels that actually have extra parameters needing to be stored'''

    pass

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
    trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode, outerIt):

    # I can loop over all the subjects and timepoints and add matrices log p(z_t | k) = log p(z_t | k, sub1,
    # tmp1) + log p(z_t | k, sub1, tmp2) + ...

    (nrSubj, nrBiomk) = crossData.shape
    nrClust = thetas.shape[0]

    dps = VoxelDPM.calcDps(subShiftsCross, crossAge1array)
    fSK = np.zeros((nrSubj, nrClust), float)
    for k in range(nrClust):
      fSK[:,k] = trajFunc(dps,thetas[k,:])

    logClustProb = np.zeros((nrBiomk,nrClust), float)
    clustProb = np.zeros((nrBiomk, nrClust), float)
    tmpSSD = np.zeros((nrBiomk, nrClust), float)
    for k in range(nrClust):
      tmpSSD[:,k] = np.sum(np.power(crossData - fSK[:,k][:, None], 2), 0) # sum across subjects, left with 1 x NR_BIOMK array
      assert(tmpSSD[:,k].shape[0] == nrBiomk)
      logClustProb[:,k] = -tmpSSD[:,k]/(2*variances[k]) - np.log(2*math.pi*variances[k])*nrSubj/2

    vertexNr = 755
    print('tmpSSD[vertexNr,:]', tmpSSD[vertexNr,:]) # good
    print('logClustProb[vertexNr,:]', logClustProb[vertexNr,:]) # bad

    for k in range(nrClust):
      expDiffs = np.power(np.e,logClustProb - logClustProb[:, k][:, None])
      clustProb[:,k] = np.divide(1, np.sum(expDiffs, axis=1))

    for c in range(nrClust):
      print('sum%d' % c, np.sum(clustProb[:,c]))

    # import pdb
    # pdb.set_trace()

    return clustProb, crossData, longData, np.nan

  def estimVariance(self, crossData, dpsCross, clustProbB, theta, nrSubjLong):
    '''
    Estimates the variance in the measurement of one biomarker. (Not variance in the mean ... )
    :param crossData: cross sectional data
    :param dpsCross: cross section disease progression scores
    :param clustProbB: clustering probabilities for one cluster only
    :param theta: parameters for
    :param nrSubjLong:
    :return: variance
    '''

    finalSSD = self.objFunTheta(theta, crossData, dpsCross, clustProbB)[1]
    # remove the degrees of freedom: 2 for each subj (slope and shift) and one for each parameters in the model
    #variance = finalSSD / (crossData.shape[0] -2*nrSubjLong - theta.shape[0])  # variance of biomarker measurement
    variance = finalSSD / (crossData.shape[0])  # variance of biomarker measurement

    return variance

  def objFunVariance(self, variance, theta, data, dpsCross, clustProbB):

    #print(subShiftsCross.shape, crossAge1array.shape, dps.shape, theta.shape, self.trajFunc(dps, theta).shape, data.shape)
    #print(test2)
    sqErrorsSB = np.power((data - self.trajFunc(dpsCross, theta)[:, None]),2)
    meanSSD = np.sum(np.multiply(clustProbB[None,:], sqErrorsSB), (0,1))

    #print("meanSSD", meanSSD, "clustProbB", clustProbB, "sqErrorsSB", sqErrorsSB)
    #print("ssd/nrSubj", meanSSD/data.shape[0])
    #print(asdsa)

    return meanSSD

  def stageSubjects(self, indices):
    # filter some diagnostic groups, i.e. PCA/tAD or for cross-validation
    filtParams = aux.filterDDSPAIndices(self.params, indices)
    filtData = filtParams['data']
    filtDiag = filtParams['diag']
    filtScanTimepts = filtParams['scanTimepts']
    filtPartCode = filtParams['partCode']
    filtAgeAtScan = filtParams['ageAtScan']

    # subjects that only contain one timepoint will be removed
    (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
      uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode) = \
      self.createLongData(filtData, filtDiag, filtScanTimepts, filtPartCode, filtAgeAtScan)

    assert(np.sum([longData[i].shape[0] for i in range(len(longData))]) == crossData.shape[0])

    longAge1array = [np.concatenate((x.reshape(-1, 1), np.ones(x.reshape(-1, 1).shape)), axis=1) for x in longAgeAtScan]

    nrSubjLong = len(longData)
    # initialise subject specific shifts and rates to the mean sub shifts from training data
    subShiftsLong = np.nan * np.ones((nrSubjLong, 2), float)
    subShiftsLong[:, 0] = np.mean(self.subShifts[:,0]) # alpha
    subShiftsLong[:, 1] = np.mean(self.subShifts[:,1]) # beta

    # estimate alphas and betas
    prevSubShiftAvg = np.mean(self.subShifts, axis=0)
    for s in range(nrSubjLong):
      subShiftsLong[s, :] = self.estimShifts(longData[s], self.thetas,
        self.variances, longAge1array[s], self.clustProb, subShiftsLong[s, :], prevSubShiftAvg,
        self.params['fixSpeed'])

    subShiftsCross = subShiftsLong[uniquePartCodeInverse,:]
    crossAge1array = np.concatenate((crossAgeAtScan.reshape(-1,1),
       np.ones(crossAgeAtScan.reshape(-1,1).shape)), axis = 1)
    dpsCross = VoxelDPM.calcDps(subShiftsCross, crossAge1array)
    clustProbBCColNorm = self.clustProb/np.sum(self.clustProb,0)[None, :]

    # fig = self.plotterObj.plotTrajWeightedDataMean(filtData, filtDiag, dpsCross, longData, longDiag, dpsLong, self.thetas,
    #     self.variances, clustProbBCColNorm, self.params['plotTrajParams'], self.trajFunc)

    print('subShiftsLong', subShiftsLong)
    print('self.subShifts', self.subShifts)

    # s = 0
    # shiftLik = self.objFunShift(subShiftsLong[s, :], longData[s], self.thetas,
    #     self.variances, longAge1array[s], self.clustProb)
    # shiftLikLo = self.objFunShift(subShiftsLong[s, :] - [0,0.5], longData[s], self.thetas,
    #     self.variances, longAge1array[s], self.clustProb)
    # shiftLikHi = self.objFunShift(subShiftsLong[s, :] + [0,0.5], longData[s], self.thetas,
    #     self.variances, longAge1array[s], self.clustProb)
    # print('shiftLik, shiftLikLo, shiftLikHi', shiftLik, shiftLikLo, shiftLikHi)

    # import pdb
    # pdb.set_trace()

    uniquePartCode = np.unique(crossPartCode)
    dpsLong = self.makeLongArray(dpsCross, scanTimepts, crossPartCode, uniquePartCode)
    maxLikStagesList = dpsLong

    nrSubjLong = len(longAgeAtScan)
    assert (len(dpsLong) == nrSubjLong)

    maxLikStages = dpsCross
    maxStagesIndex = None
    stagingProb = None
    stagingLik = None
    tsStages = None
    otherParams = dict(subShiftsLong=subShiftsLong, subShiftsCross=subShiftsCross,
      longPartCode=longPartCode, longAgeAtScan=longAgeAtScan, longDiag=longDiag)

    return maxLikStages, maxStagesIndex, stagingProb, stagingLik, maxLikStagesList, otherParams



  def stageSubjectsIndep(self, indices):
    # filter some diagnostic groups, i.e. PCA/tAD or even for cross-validation
    filtParams = aux.filterDDSPAIndices(self.params, indices)
    filtData = filtParams['data']
    filtDiag = filtParams['diag']
    filtScanTimepts = filtParams['scanTimepts']
    filtPartCode = filtParams['partCode']
    filtAgeAtScan = filtParams['ageAtScan']

    # subjects that only contain one timepoint will be removed
    (longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan,
      uniquePartCodeInverse, crossData, crossDiag, scanTimepts, crossPartCode, crossAgeAtScan, uniquePartCode) = \
      self.createLongData(filtData, filtDiag, filtScanTimepts, filtPartCode, filtAgeAtScan)

    assert(np.sum([longData[i].shape[0] for i in range(len(longData))]) == crossData.shape[0])

    longAge1array = [np.concatenate((x.reshape(-1, 1), np.ones(x.reshape(-1, 1).shape)), axis=1) for x in longAgeAtScan]
    crossAgeZeros1array = np.concatenate((np.zeros(crossAgeAtScan.shape).reshape(-1,1),
                                          np.ones(crossAgeAtScan.reshape(-1,1).shape)), axis = 1)

    nrSubjCross = len(crossData)
    # initialise subject specific shifts and rates to the mean sub shifts from training data
    subShiftsCross = np.zeros((nrSubjCross, 2), float)
    subShiftsCross[:, 0] = 1 # alpha
    subShiftsCross[:, 1] = np.mean(self.dpsLongFirstVisit) - crossAgeAtScan # beta

    # estimate alphas and betas
    prevSubShiftAvg = np.mean(self.subShifts, axis=0)
    for s in range(nrSubjCross):
      subShiftsCross[s, :] = self.estimShifts(crossData[s,:].reshape(1,-1), self.thetas,
        self.variances, crossAgeZeros1array[s,:].reshape(1,-1), self.clustProb, subShiftsCross[s, :],
        prevSubShiftAvg, fixSpeed=True)

    dpsCross = VoxelDPM.calcDps(subShiftsCross, crossAgeZeros1array)
    clustProbBCColNorm = self.clustProb/np.sum(self.clustProb,0)[None, :]

    # fig = self.plotterObj.plotTrajWeightedDataMean(filtData, filtDiag, dpsCross, longData, longDiag, dpsLong, self.thetas,
    #     self.variances, clustProbBCColNorm, self.params['plotTrajParams'], self.trajFunc)

    # import pdb
    # pdb.set_trace()

    uniquePartCode = np.unique(crossPartCode)

    dpsLong = self.makeLongArray(dpsCross, scanTimepts, crossPartCode, uniquePartCode)

    nrSubjLong = len(longAgeAtScan)
    assert(len(dpsLong) == nrSubjLong)

    # plotTrajParams = self.params['plotTrajParams']
    # fig = pl.figure()
    # for s in range(nrSubjLong):
    #   pl.plot(longAgeAtScan[s], dpsLong[s], '%s-' % (plotTrajParams['diagColors'][longDiag[s]-1]))
    #
    # pl.show()
    # print(adsdas)

    maxLikStagesList = dpsLong


    return maxLikStagesList, longAgeAtScan, longDiag

  @staticmethod
  def calcDps(subShifts, age1array):
    """calculated the disease progresison score from subject shifts and age. should work for both Cross and Long"""

    dps = np.sum(np.multiply(subShifts, age1array), 1)

    return dps

  @staticmethod
  def calcDpsNo1array(subShifts, ageAtScan):
    """calculated the disease progresison score from subject shifts and age. should work for both Cross and Long"""
    age1array = np.concatenate((ageAtScan.reshape(-1, 1),
    np.ones(ageAtScan.reshape(-1, 1).shape)), axis=1)
    dps = VoxelDPM.calcDps(subShifts, age1array)

    return dps


  def getFittedParams(self):
    return [self.thetas, self.variances, self.subShifts, self.clustProb]


  def stageSubjectsData(self, data):
    raise NotImplementedError("Should have implemented this")

  def stageSubjectsCrossDataAge(self, data, age):
    # filter some diagnostic groups, i.e. PCA/tAD or for cross-validation

    # data = data[:20,:]
    # age = age[:20]

    crossAge1array = np.concatenate((age.reshape(-1, 1), np.ones(age.reshape(-1, 1).shape)), axis=1)

    nrSubjCross = data.shape[0]
    # initialise subject specific shifts and rates to the mean sub shifts from training data
    subShiftsCross = np.nan * np.ones((nrSubjCross, 2), float)
    subShiftsCross[:, 0] = np.mean(self.subShifts[:,0]) # alpha
    subShiftsCross[:, 1] = np.mean(self.subShifts[:,1]) # beta

    # estimate alphas and betas
    prevSubShiftAvg = np.mean(self.subShifts, axis=0)
    for s in range(nrSubjCross):
      print('part %d/%d' % (s, nrSubjCross))
      subShiftsCross[s, :] = self.estimShifts(data[s,:], self.thetas,
        self.variances, crossAge1array[s,:].reshape(1,-1), self.clustProb, subShiftsCross[s, :], prevSubShiftAvg,
        self.params['fixSpeed'])

    dpsCross = VoxelDPM.calcDps(subShiftsCross, crossAge1array)

    longData = [data[s, :].reshape(1,-1) for s in range(nrSubjCross)]
    longDiag = [1 for s in range(nrSubjCross)]
    dpsLong = [dpsCross[s] for s in range(nrSubjCross)]

    # clustProbBCColNorm = self.clustProb/np.sum(self.clustProb,0)[None, :]
    # fig = self.plotterObj.plotTrajWeightedDataMean(data, np.ones(data.shape[0]), dpsCross,
    #   longData, longDiag, dpsLong, self.thetas, self.variances, clustProbBCColNorm,
    #   self.params['plotTrajParams'], self.trajFunc, replaceFigMode=False)

    # print('subShiftsCross', subShiftsCross)
    # print('self.subShifts', self.subShifts)

    # print(adsa)


    # import pdb
    # pdb.set_trace()


    return subShiftsCross, dpsCross


  def plotTrajectories(self, res):
    ''' this is already done in function postFitAnalysis() '''
    pass

  def plotTrajSummary(self, res):
    raise NotImplementedError("Should have implemented this")

  def trajFunc(self, s, theta):
    """
    sigmoidal function for trectory with params [a,b,c,d] with
    minimum d, maximum a+d, slope a*b/4 and slope
    maximum attained at center c
    f(s|theta = [a,b,c,d]) = a/(1+exp(-b(s-c)))+d

    :param s: the inputs and can be an array of dim N x 1
    :param theta: parameters as np.array([a b c d])
    :return: values of the sigmoid function at the inputs s
    """
    # print('theta',theta)
    # print(s - theta[2])
    # print(np.exp(-theta[1] * (s - theta[2])))
    # print(np.power((1 + np.exp(-theta[1] * (s - theta[2]))), -1))
    return theta[0] * np.power((1 + np.exp(-theta[1] * (s - theta[2]))), -1) + theta[3]

  def runInitClust(self, runPart, crossData, crossDiag):
    # printStats(longData, longDiag)
    sys.stdout.flush()
    np.random.seed(1)
    plotTrajParams = self.params['plotTrajParams']
    initClustFile = '%s/initClust.npz' % self.outFolder
    nrClust = self.params['nrClust']
    if runPart[0] == 'R':
      # perform some data driven clustering in order to get some initial clustering probabilities

      if self.params['initClustering'] == 'k-means':
        # perform k-means usign scikit-learn
        initClustSubsetInd = self.params['initClustSubsetInd']
        nearNeighInitClust = self.params['nearNeighInitClust']

        print('CROSS DATA',crossData[:,initClustSubsetInd].T)
        print(nrClust)

        clustResStruct = sklearn.cluster.KMeans(n_clusters=nrClust, random_state=0)\
          .fit(crossData[:,initClustSubsetInd].T)

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
        assert (all(np.mean(crossData[crossDiag == CTL], axis=0) < 0.5))
        avgCrossDataPatB = np.mean(crossData[crossDiag == self.params['patientID']], axis=0)
        percentiles = list(np.percentile(avgCrossDataPatB,
          [c * 100 / nrClust for c in range(1, nrClust)]))
        percentiles = [-float("inf")] + percentiles + [float("inf")]
        assert (len(percentiles) == (nrClust + 1))
        initClust = np.zeros(crossData.shape[1], int)
        for c in range(nrClust):
          clustMask = np.logical_and(percentiles[c] < avgCrossDataPatB,
            avgCrossDataPatB < percentiles[c + 1])
          initClust[clustMask] = c

      elif self.params['initClustering'] == 'fsurf':
        fsaverageAnnotFile = '%s/subjects/fsaverage/label/lh.aparc.annot' % \
                             plotTrajParams['freesurfPath']
        labels, ctab, names = nib.freesurfer.io.read_annot(fsaverageAnnotFile, orig_ids=False)

        print(labels, ctab, [names[i].decode('utf-8') for i in range(len(names))])
        print(labels.shape, ctab.shape, len(names))

        assert (nrClust in [3,4,5,8,20,36])

        if nrClust == 36:
          assert nrClust == len(names)
          # map each ROI in the annotation for a cluster number in range (0, nrClust-1)
          # clust 0-frontal   1-parietal   2-occipital   3-temporal  4-cingulate+other
          assert (any(labels == 0))
          print(np.min(labels), np.max(labels))
          initClust = labels[plotTrajParams['pointIndices']]
          assert initClust.shape[0] == crossData.shape[1]
          print(np.min(initClust), np.max(initClust))
          # print(initClust[:1000])
          for c in range(-1, nrClust + 5):
            print(c, np.sum(initClust == c))
          initClust[initClust < 0] = 0  # allocate the -1s to 0
          # initClust -= 1 # make them start from 0
          assert (np.max(initClust) == (nrClust - 1))
          # print(np.max(initClust))
          # print(asdsd)
          # print(finClust36)
        else:
          if nrClust == 3:
            # map each ROI in the annotation for a cluster number in range (0, nrClust-1)
            # clust 0-frontal   1-parietal+occipital   2-temporal
            mapFSNamestoClusts = {'unknown': 0, 'bankssts': 2, 'caudalanteriorcingulate': 1, 'caudalmiddlefrontal': 0,
              'corpuscallosum': 1, 'cuneus': 1, 'entorhinal': 2, 'fusiform': 2, 'inferiorparietal': 1,
              'inferiortemporal': 2, 'isthmuscingulate': 2, 'lateraloccipital': 1, 'lateralorbitofrontal': 0,
              'lingual': 1, 'medialorbitofrontal': 0, 'middletemporal': 2, 'parahippocampal': 2, 'paracentral': 2,
              'parsopercularis': 0, 'parsorbitalis': 0, 'parstriangularis': 0, 'pericalcarine': 1, 'postcentral': 1,
              'posteriorcingulate': 1, 'precentral': 0, 'precuneus': 1, 'rostralanteriorcingulate': 0,
              'rostralmiddlefrontal': 0, 'superiorfrontal': 0, 'superiorparietal': 1, 'superiortemporal': 2,
              'supramarginal': 1, 'frontalpole': 0, 'temporalpole': 2, 'transversetemporal': 2, 'insula': 2}

          if nrClust == 4:
            # map each ROI in the annotation for a cluster number in range (0, nrClust-1)
            # clust 0-frontal   1-parietal   2-occipital   3-temporal
            mapFSNamestoClusts = {'unknown': 0, 'bankssts': 3, 'caudalanteriorcingulate': 2, 'caudalmiddlefrontal': 0,
              'corpuscallosum': 1, 'cuneus': 2, 'entorhinal': 3, 'fusiform': 3, 'inferiorparietal': 1,
              'inferiortemporal': 3, 'isthmuscingulate': 3, 'lateraloccipital': 2, 'lateralorbitofrontal': 0,
              'lingual': 2, 'medialorbitofrontal': 0, 'middletemporal': 3, 'parahippocampal': 3, 'paracentral': 3,
              'parsopercularis': 0, 'parsorbitalis': 0, 'parstriangularis': 0, 'pericalcarine': 2, 'postcentral': 1,
              'posteriorcingulate': 2, 'precentral': 0, 'precuneus': 1, 'rostralanteriorcingulate': 0,
              'rostralmiddlefrontal': 0, 'superiorfrontal': 0, 'superiorparietal': 1, 'superiortemporal': 3,
              'supramarginal': 1, 'frontalpole': 0, 'temporalpole': 3, 'transversetemporal': 3, 'insula': 3}

          if nrClust == 5:
            # map each ROI in the annotation for a cluster number in range (0, nrClust-1)
            # clust 0-frontal   1-parietal   2-occipital   3-temporal  4-cingulate+other
            mapFSNamestoClusts = {'unknown': 4, 'bankssts': 4, 'caudalanteriorcingulate': 4, 'caudalmiddlefrontal': 0,
              'corpuscallosum': 4, 'cuneus': 2, 'entorhinal': 3, 'fusiform': 3, 'inferiorparietal': 1,
              'inferiortemporal': 3, 'isthmuscingulate': 4, 'lateraloccipital': 2, 'lateralorbitofrontal': 0,
              'lingual': 2, 'medialorbitofrontal': 0, 'middletemporal': 3, 'parahippocampal': 3, 'paracentral': 3,
              'parsopercularis': 0, 'parsorbitalis': 0, 'parstriangularis': 0, 'pericalcarine': 2, 'postcentral': 1,
              'posteriorcingulate': 4, 'precentral': 0, 'precuneus': 1, 'rostralanteriorcingulate': 4,
              'rostralmiddlefrontal': 0, 'superiorfrontal': 0, 'superiorparietal': 1, 'superiortemporal': 3,
              'supramarginal': 1, 'frontalpole': 0, 'temporalpole': 3, 'transversetemporal': 3, 'insula': 4}

          if nrClust == 8:
            # map each ROI in the annotation for a cluster number in range (0, nrClust-1)
            # clust 0-sup frontal   1-inf front    2-sup parietal    3-inf parietal
            # 4-occipital   5-sup temporal    6-inf temporal    7-cingulate+other
            mapFSNamestoClusts = {'unknown': 7, 'bankssts': 7, 'caudalanteriorcingulate': 7, 'caudalmiddlefrontal': 0,
              'corpuscallosum': 7, 'cuneus': 4, 'entorhinal': 7, 'fusiform': 6, 'inferiorparietal': 3,
              'inferiortemporal': 6, 'isthmuscingulate': 7, 'lateraloccipital': 4, 'lateralorbitofrontal': 0,
              'lingual': 7, 'medialorbitofrontal': 0, 'middletemporal': 5, 'parahippocampal': 7, 'paracentral': 5,
              'parsopercularis': 0, 'parsorbitalis': 0, 'parstriangularis': 0, 'pericalcarine': 4, 'postcentral':2,
              'posteriorcingulate': 7, 'precentral': 0, 'precuneus': 3, 'rostralanteriorcingulate': 7,
              'rostralmiddlefrontal': 0, 'superiorfrontal': 0, 'superiorparietal': 2, 'superiortemporal': 5,
              'supramarginal': 2, 'frontalpole': 0, 'temporalpole': 6, 'transversetemporal': 5, 'insula': 7}

          if nrClust == 20:
            # map each ROI in the annotation for a cluster number in range (0, nrClust-1)
            # clust 0-other
            mapFSNamestoClusts = {'unknown': 0, 'bankssts': 0, 'caudalanteriorcingulate': 1, 'caudalmiddlefrontal': 2,
              'corpuscallosum': 3, 'cuneus': 4, 'entorhinal': 5, 'fusiform': 6, 'inferiorparietal': 7,
              'inferiortemporal': 8, 'isthmuscingulate': 1, 'lateraloccipital': 9, 'lateralorbitofrontal': 10,
              'lingual': 11, 'medialorbitofrontal': 12, 'middletemporal': 13, 'parahippocampal': 5, 'paracentral': 14,
              'parsopercularis': 15, 'parsorbitalis': 15, 'parstriangularis': 15, 'pericalcarine': 9, 'postcentral':16,
              'posteriorcingulate': 1, 'precentral': 17, 'precuneus': 18, 'rostralanteriorcingulate': 1,
              'rostralmiddlefrontal': 10, 'superiorfrontal': 17, 'superiorparietal': 18, 'superiortemporal': 19,
              'supramarginal': 18, 'frontalpole': 10, 'temporalpole': 8, 'transversetemporal': 8, 'insula': 0}

          fsNr2ClustNrMap = np.array([mapFSNamestoClusts[n.decode('utf-8')] for n in names], int)
          assert (len(fsNr2ClustNrMap) == len(names))
          assert (any(labels == 0))
          print(fsNr2ClustNrMap[labels].shape, fsNr2ClustNrMap[:10])
          # print(dsa)
          initClust = fsNr2ClustNrMap[labels][plotTrajParams['pointIndices']]
          assert initClust.shape[0] == crossData.shape[1]

      elif self.params['initClustering'] == 'noClust':
        initClust = range(nrClust)
      elif self.params['initClustering'] == 'predefined':
        initClust = self.params['initClust']
      else:
        raise ValueError('--initClustering needs to be either k-means or hist')

      os.system('mkdir -p %s' % self.outFolder)
      clustDataStruct = dict(initClust=initClust)
      pickle.dump(clustDataStruct, open(initClustFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    elif runPart[0] == 'L':
      clustDataStruct = pickle.load(open(initClustFile, 'rb'))
      initClust = clustDataStruct['initClust']
      assert (np.min(initClust) == 0)

    elif runPart[0] == 'Non-enforcing':
      if os.path.isfile(initClustFile):
        clustDataStruct = pickle.load(open(initClustFile, 'rb'))
        initClust = clustDataStruct['initClust']
      else:
        print('no file found at runPart[0] for %s' % self.outFolder)
        #initClust = np.zeros(crossData.shape[1], int)
        return None
    else:
      raise ValueError('runPart needs to be either R, L or Non-enforcing')

    return initClust

  def postFitAnalysis(self, runPart, crossData, crossDiag, dpsCross, clustProbBCColNorm,
                      paramsDataFile, resStruct):
    nrClust = self.params['nrClust']
    plotTrajParams = self.params['plotTrajParams']
    nrSubjLong = self.subShifts.shape[0]

    # returns a hue value for each cluster, in the same cluster order as clustProbBC or thetas
    hueColsAtrophyExtentC, rgbColsAtrophyExtentC, sortedIndAtrophyExtent = \
      self.plotterObj.colorClustByAtrophyExtent(self.thetas, dpsCross, crossDiag,
        self.params['patientID'], self.trajFunc)

    if runPart[3] == 'R':

      doCogCorr = False

      if doCogCorr:
        _, _, _, dpsCrossNorm, _, _= \
          VoxelDPM.normDPS(resStruct['subShifts'], resStruct['ageFirstVisitLong1array'],
            resStruct['longDiag'], resStruct['uniquePartCodeInverse'],
            resStruct['crossAge1array'], resStruct['thetas'], None)

        testLabels = ['CDRSOB', 'ADAS13', 'MMSE', 'RAVLT']
        nrTests = len(testLabels)
        inversionFactors = np.array([1,1,-1,-1])

        cogTests = self.params['cogTests'] * inversionFactors[None, :]


        corrAllTestData = [0 for x in range(nrTests)]
        pValAllTestData = [0 for x in range(nrTests)]

        from sklearn import linear_model

        for t in range(nrTests):
          # for d in range(nrUnqDiag):
          #   pl.scatter(dpsCrossNorm[longDiag == diagNrs[d]], cogTestsTest[longDiag == diagNrs[d],t])
          fig = pl.figure()
          notNanMask = np.logical_not(np.isnan(cogTests[:,t]))

          pl.scatter(dpsCrossNorm, cogTests[:, t])
          regr = linear_model.LinearRegression()
          regr.fit(dpsCrossNorm[notNanMask].reshape(-1,1), cogTests[notNanMask, t].reshape(-1,1))
          predictedVals = regr.predict(dpsCrossNorm[notNanMask].reshape(-1,1)).reshape(-1)

          corrAllTestData[t], pValAllTestData[t] = scipy.stats.pearsonr(
            dpsCrossNorm[notNanMask].reshape(-1, 1), cogTests[notNanMask, t].reshape(-1, 1))

          pl.plot(dpsCrossNorm[notNanMask].reshape(-1),predictedVals, color='k', linewidth=3)

          fs = 18
          pl.xlabel('Disease Progression Score (DPS)',fontsize=fs)
          pl.ylabel(testLabels[t],fontsize=fs)
          pl.xticks(fontsize=fs)
          pl.yticks(fontsize=fs)
          pl.gcf().subplots_adjust(bottom=0.15,left=0.15)

          fig.show()
          fig.savefig('%s/stagingCogTestsScatterPlot_%s_%s.png' % (plotTrajParams['outFolder'],
            self.expName, testLabels[t]), dpi=100)


        print('corrAllTestData', corrAllTestData)
        print('pValAllTestData', pValAllTestData)
        print(testLabels)
        # print(ads)

      self.plotterObj.makeMovie(self.thetas, dpsCross, crossDiag, self.trajFunc, self.clustProb, plotTrajParams,
          filePathNoExt='%s/movie_%s' % (self.outFolder, '_'.join(self.expName.split('/'))))

      # self.plotterObj.makeSnapshots(self.thetas, dpsCross, crossDiag, self.trajFunc, self.clustProb, plotTrajParams,
      #                           filePathNoExt='%s/snapshots_%s' % (self.outFolder, '_'.join(self.expName.split('/'))))

      # self.plotterObj.plotBlenderDirectGivenClustCols(hueColsAtrophyExtentC, self.clustProb, plotTrajParams,
      #   filePathNoExt='%s/atrophyExtent_%s' % (self.outFolder, '_'.join(self.expName.split('/'))))

      # print(adsa)

    # if self.params['patientID'] == AD:
    thetaSamplesFile = '%s/samplesTheta.npz' % self.outFolder

    nrParamsTheta = self.thetas.shape[1]

    if runPart[4] == 'R':

      assert(all(np.abs(np.sum(self.clustProb,1) - 1) < 0.001) )

      # sample trajectories with MCMC for each cluster and perform t-test to see if the differences are statistically different

      nrSamples = 300
      thetasSamples = np.zeros((nrClust, nrSamples, nrParamsTheta), float)
      variancesSamples = np.zeros((nrClust, nrSamples), float)

      thetaVariances = np.var(self.thetas, axis=0)
      propCovMat = np.diag([0, thetaVariances[1], thetaVariances[2], 0, np.var(self.variances)])/2000

      # ------------------------ perform sampling of thetas -------------------
      for c in range(nrClust):
        (thetasSamples[c,:,:], variancesSamples[c,:]) = self.sampleThetas(
          crossData, dpsCross, clustProbBCColNorm[:,c], self.thetas[c, :], self.variances[c],
          nrSubjLong, nrSamples, propCovMat)
      # only take every 10 samples to make them more independent
      thetasSamples = thetasSamples[:,::10,:]
      variancesSamples = variancesSamples[:,::10]
      print(thetasSamples.shape, variancesSamples.shape)
      dataStruct = dict(thetasSamples=thetasSamples,variancesSamples=variancesSamples)
      pickle.dump(dataStruct, open(thetaSamplesFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
      # -----------------------------------------------------------------------

    elif runPart[4] == 'L':
      dataStruct = pickle.load(open(thetaSamplesFile, 'rb'))
      thetasSamples = dataStruct['thetasSamples']
      variancesSamples = dataStruct['variancesSamples']
      print('thetasSamples.shape', thetasSamples.shape)

    if runPart[4] == 'R' or runPart[4] == 'L':
      # perform t-test between each pair of clusters
      pVals = np.zeros((nrClust, nrClust, nrParamsTheta), float)
      for c1 in range(nrClust):
        for c2 in range(c1+1,nrClust):
          for p in range(nrParamsTheta):
            tstat, pVals[c1,c2,p] = scipy.stats.ttest_ind(thetasSamples[c1,:,p],
              thetasSamples[c2,:,p], axis = 0, equal_var = True)

      # only interested in slope and timeshift for sigmoid
      print('pVals[:,:,1] ', pVals[:,:,1], 'pVals[:,:,2]', pVals[:,:,2])

      # fig = self.plotterObj.plotTrajWeightedDataMean(crossData, crossDiag, dpsCross, longData, longDiag, dpsLong, self.thetas, self.variances,
      #                                clustProbBCColNorm, self.params['plotTrajParams'], self.trajFunc,
      #                                thetasSamplesClust=thetasSamplesClust, replaceFigMode=False, showConfInt=False,
      #                                colorTitle=False,
      #                                yLimUseData=True, adjustBottomHeight=0.175, orderClust=True)
      # fig.savefig('%s/trajSamplesMeanData_%s.png' % (self.outFolder,
      #    self.outFolder.split('/')[-1]), dpi = 100)


      # dataStruct = dict(crossData=crossData, crossDiag=crossDiag, dpsCross=dpsCross, thetas=self.thetas,
      #   variances=self.variances, clustProbBCColNorm=clustProbBCColNorm, plotTrajParams=self.params['plotTrajParams'],
      #   trajFunc=self.trajFunc)

      fig = self.plotterObj.plotTrajSamplesInOneNoData(crossData, crossDiag, dpsCross, self.subShifts,
        resStruct['ageFirstVisitLong1array'], resStruct['longDiag'], resStruct['uniquePartCodeInverse'],
        resStruct['crossAge1array'], self.thetas, self.variances, clustProbBCColNorm,
        self.params['plotTrajParams'], self.trajFunc, thetasSamples=thetasSamples,
        replaceFigMode = True, showConfInt=False,colorTitle=False, yLimUseData=True, windowSize=(500,400),
        clustColsHues=hueColsAtrophyExtentC)

      fig.savefig('%s/trajSamplesOneFig_%s.png' % (self.outFolder,
         '_'.join(self.expName.split('/'))), dpi = 100)

      # print(adsas)

      # fig = self.plotterObj.plotTrajSamplesHist(crossData, crossDiag, dpsCross, self.thetas, self.variances,
      #   clustProbBCColNorm, self.params['plotTrajParams'], self.trajFunc,
      #   thetasSamplesClust=thetasSamplesClust, replaceFigMode = True, showConfInt=False,colorTitle=False,
      #   yLimUseData=True, adjustBottomHeight=0.175, orderClust=True, windowSize=(500,400))
      # fig.savefig('%s/trajSamplesPlusHist_%s.png' % (self.outFolder,
      #    '_'.join(self.expName.split('/'))), dpi = 100)


  def exponClustProb(self, logClustProbBC):
    # renormalise the clust probabilities
    # if probs are too low, truncate them to a v. small number
    logClustProbBC[logClustProbBC < -100] = -100 # in log space this is a v. small number
    clustProb = np.zeros(logClustProbBC.shape, float)
    for k in range(logClustProbBC.shape[1]):
      expDiffs = np.power(np.e, logClustProbBC - logClustProbBC[:, k][:, None])
      clustProb[:, k] = np.divide(1, np.sum(expDiffs, axis=1))

    return clustProb


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

    assert(np.max(inverseMap) == len(longData)-1) # inverseMap indices should be smaller than the size of longData as they take elements from longData
    assert(len(inverseMap) == filtData.shape[0]) # length of inversemap should be the same as the cross-sectional data

    #print(np.max(inverseMap), len(longData), len(inverseMap), inverseMap.shape)
    #print(test)

    return longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan, \
      inverseMap, filtData, filtDiag, filtScanTimetps, filtPartCode, filtAgeAtScan, uniquePartCode


  def makeLongArray(self, array, scanTimepts, partCode, uniquePartCode):
    # place data in a longitudinal format
    longArray = [] # longArray can be data, diag, ageAtScan,scanTimepts, etc .. both 1D or 2D
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
      print('Current Part Time', len(currPartTimeptsOrdInd))
      #assert(len(currPartTimeptsOrdInd) >= 2) # 2 for PET, 3 for MRI

      if len(currPartTimeptsOrdInd) > 1:
        longArray += [array[currPartIndicesOrd]]

    return longArray

  def inferMissingData(self, crossData,longData, prevClustProbBC, thetas, subShiftsCross,
  crossAge1array, trajFunc, scanTimepts, partCode, uniquePartCode, plotterObj):

    ''' don't do anything, only used by the VDPMNan model '''

    return crossData, longData
