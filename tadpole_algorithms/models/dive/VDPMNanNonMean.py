import voxelDPM
# import VDPMMean
import numpy as np
import scipy
import sys
from env import *
import os
import pickle
import sklearn
import math
import numpy.ma as ma
import VDPMNan


''' Class for a Voxelwise Disease Progression Model that can handle missing data (NaNs).
    uses masked arrays for fitting the model (hence no data inference in E-step)

    VDPM for NaNs that doesn't use the fast implementation, which created some problems
    as it biased the SSD calculation, since when there was missing data corresponding to a high
    clustering probability, the error was dominated by the other (present) biomkarker data that
    had correspondingly low clustering probabilities for that particular cluster

'''

class VDPMNanNonMeanBuilder(voxelDPM.VoxelDPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMNanNonMean(dataIndices, expName, params, self.plotterObj)

class VDPMNanNonMean(VDPMNan.VDPMNan):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)

    self.nanMask = np.nan


  def runInitClust(self, runPart, crossData, crossDiag):
    # printStats(longData, longDiag)
    nrClust = self.params['nrClust']

    os.system('mkdir -p %s' % self.outFolder)
    initClust = np.array(range(nrClust))
    assert nrClust == crossData.shape[1]

    return initClust

  def inferMissingData(self, crossData,longData, prevClustProbBC, thetas, subShiftsCross,
    crossAge1array, trajFunc, scanTimepts, partCode, uniquePartCode, plotterObj):

    ''' don't do anything, leave data with NaNs in this model! '''

    self.nanMask = np.isnan(crossData)
    plotterObj.nanMask = self.nanMask
    plotterObj.longDataNaNs = longData

    crossDataMasked = np.ma.masked_array(crossData, np.isnan(crossData))
    longDataMasked = [np.ma.masked_array(d, np.isnan(d))
      for d in longData]

    return crossDataMasked, longDataMasked

  # def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
  #   trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode):
  #   # overwrite function as we need to use a different variance (in the biomk measurements as opposed to their mean)
  #   return prevClustProbBC, crossData, longData

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
    trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode, outerIt):
    # overwrite function as we need to use a different variance (in the biomk measurements as opposed to their mean)

    # I can loop over all the subjects and timepoints and add matrices log p(z_t | k) = log p(z_t | k, sub1,
    # tmp1) + log p(z_t | k, sub1, tmp2) + ...

    (nrSubj, nrBiomk) = crossData.shape
    nrClust = thetas.shape[0]
    nrSubjWithDataPerBiomkB = np.sum(np.logical_not(np.isnan(crossData)),axis=0)

    # print('nrSubjWithDataPerBiomkB', nrSubjWithDataPerBiomkB)
    # print(adsa)

    dps = voxelDPM.VoxelDPM.calcDps(subShiftsCross, crossAge1array)
    fSK = np.zeros((nrSubj, nrClust), float)
    for k in range(nrClust):
      fSK[:,k] = trajFunc(dps,thetas[k,:])

    logClustProb = np.zeros((nrBiomk,nrClust), np.longdouble)
    clustProb = np.zeros((nrBiomk, nrClust), float)
    tmpSSD = np.zeros((nrBiomk, nrClust), np.longdouble)
    for k in range(nrClust):
      tmpSSD[:,k] = np.nansum(np.power(crossData - fSK[:,k][:, None], 2), 0) # sum across subjects, left with 1 x NR_BIOMK array
      assert(tmpSSD[:,k].shape[0] == nrBiomk)
      logClustProb[:,k] = -tmpSSD[:,k]/(2*variances[k]) - np.log(2*math.pi*variances[k])*nrSubjWithDataPerBiomkB/2

    # vertexNr = 755
    # print('tmpSSD[vertexNr,:]', tmpSSD[vertexNr,:]) # good
    # print('logClustProb[vertexNr,:]', logClustProb[vertexNr,:]) # bad

    for k in range(nrClust):
      expDiffs = np.power(np.e,logClustProb - logClustProb[:, k][:, None])
      clustProb[:,k] = np.divide(1, np.sum(expDiffs, axis=1))

    for c in range(nrClust):
      print('sum%d' % c, np.sum(clustProb[:,c]))


    print('clustProb', clustProb)
    print('nan entries biomk', np.sum(np.isnan(clustProb),axis=0) > 0)
    print('nan entries clust', np.sum(np.isnan(clustProb),axis=1) > 0)
    if np.isnan(clustProb).any():
      print('error, NaN entries in clustProb')
      import pdb
      pdb.set_trace()

    return clustProb, crossData, longData, np.nan


  def estimShifts(self, dataOneSubjTB, thetas, variances, ageOneSubj1array, clustProbBC,
    prevSubShift, prevSubShiftAvg, fixSpeed):

    '''
    do not use dot product because when NaNs are involved the weights will not sum to 1.
    use np.ma.average(.., weights) instead, as the weights will be re-normalised accordingly
    '''

    # print('prevSubShift, prevSubShiftAvg', prevSubShift, prevSubShiftAvg)
    # print(adsa)

    clustProbBCColNorm = clustProbBC / np.sum(clustProbBC, 0)[None, :]

    nrBiomk, nrClust = clustProbBC.shape
    nrTimepts = dataOneSubjTB.shape[0]

    dataOneSubjTBarray = np.array(dataOneSubjTB)
    dataOneSubjTBarray[dataOneSubjTB.mask] = np.nan

    if fixSpeed: # fixes parameter alpha to 1
      composeShift = lambda beta: [prevSubShiftAvg[0], beta]
      initSubShift = prevSubShift[1]
      print('prevSubShift', prevSubShift)
      print('initSubShift', initSubShift)
      # asda
      objFuncLambda = lambda beta: self.objFunShift(composeShift(beta), dataOneSubjTBarray, thetas,
        variances, ageOneSubj1array, clustProbBC)

      prevSubShiftAvgCurr = prevSubShiftAvg[1].reshape(1,-1)
    else:
      composeShift = lambda shift: shift
      initSubShift = prevSubShift
      objFuncLambda = lambda beta: self.objFunShift(composeShift(beta), dataOneSubjTBarray, thetas,
        variances, ageOneSubj1array, clustProbBC)

      prevSubShiftAvgCurr = prevSubShiftAvg

    # print('composeShift(initSubShift)', composeShift(initSubShift))
    # print('objFuncLambda(initSubShift)', objFuncLambda(initSubShift))
    # print(ads)

    res = scipy.optimize.minimize(objFuncLambda, initSubShift, method='Nelder-Mead',
                                  options={'xatol': 1e-2, 'disp': False})
    bestShift = res.x
    nrStartPoints = 2
    nrParams = prevSubShiftAvgCurr.shape[0]
    pertSize = 1
    minSSD = res.fun
    success = False
    for i in range(nrStartPoints):
      perturbShift = prevSubShiftAvgCurr * (np.ones(nrParams) + pertSize *
        np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
      res = scipy.optimize.minimize(objFuncLambda, perturbShift, method='Nelder-Mead',
        options={'xtol': 1e-8, 'disp': False, 'maxiter': 100})
      currShift = res.x
      currSSD = res.fun
      # print('currSSD', currSSD, objFuncLambda(currShift))
      if currSSD < minSSD:
        # if we found a better solution then we decrease the step size
        minSSD = currSSD
        bestShift = currShift
        pertSize /= 1.2
        success = res.success
      else:
        # if we didn't find a solution then we increase the step size
        pertSize *= 1.2
    # print('bestShift', bestShift)

    return composeShift(bestShift)

  def objFunShift(self, shift, dataOneSubj, thetas, variances, ageOneSubj1array, clustProb):

    # print('shift, ageOneSubj1array', shift, ageOneSubj1array)
    dps = np.sum(np.multiply(shift, ageOneSubj1array),1)
    nrClust = clustProb.shape[1]

    # print('shift', shift)
    # print('ageOneSubj1array', ageOneSubj1array)
    # print('dps', dps)
    # asda
    sumSSD = 0
    for k in range(nrClust):
      sqErrorsB = np.nansum(np.power((dataOneSubj - self.trajFunc(dps, thetas[k,:])[:, None]),2), axis=0)
      sumSSD += np.nansum(np.multiply(sqErrorsB, clustProb[:,k]))/(2*variances[k])

    logPriorShift = self.logPriorShiftFunc(shift, self.paramsPriorShift)

    # print('logPriorShift', logPriorShift, 'sumSSD', sumSSD)
    # print(sumSSD)
    # if shift[0] < -400: # and -67
    #   import pdb
    #   pdb.set_trace()

    return sumSSD - logPriorShift


  def estimThetas(self, data, dpsCross, clustProbColNormB, prevTheta, nrSubjLong, prevThetas):
    '''
    data contains NaNs.
    '''

    recompThetaSig = lambda thetaFull, theta12: [thetaFull[0], theta12[0], theta12[1], thetaFull[3]]

    # print('estimThetas data shape', data.shape)
    dataNpArray = np.array(data)
    dataNpArray[data.mask] = np.nan

    objFuncLambda = lambda theta12: self.objFunTheta(recompThetaSig(prevTheta, theta12),
      dataNpArray, dpsCross, clustProbColNormB)[0]

    initTheta12 = prevTheta[[1, 2]]

    nrStartPoints = 10
    nrParams = initTheta12.shape[0]
    pertSize = 1
    minTheta = np.array([-1/np.std(dpsCross), -np.inf])
    maxTheta = np.array([0, np.inf])
    minSSD = np.inf
    bestTheta = initTheta12
    success = False
    for i in range(nrStartPoints):
      perturbTheta = initTheta12 * (np.ones(nrParams) + pertSize *
        np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
      # print('perturbTheta < minTheta', perturbTheta < minTheta)
      # perturbTheta[perturbTheta < minTheta] = minTheta[perturbTheta < minTheta]
      # perturbTheta[perturbTheta > maxTheta] = minTheta[perturbTheta > maxTheta]
      res = scipy.optimize.minimize(objFuncLambda, perturbTheta, method='Nelder-Mead',
        options={'xtol': 1e-8, 'disp': True, 'maxiter':100})
      currTheta = res.x
      currSSD = res.fun
      print('currSSD', currSSD, objFuncLambda(currTheta))
      if currSSD < minSSD:
        # if we found a better solution then we decrease the step size
        minSSD = currSSD
        bestTheta = currTheta
        pertSize /= 1.2
        success = res.success
      else:
        # if we didn't find a solution then we increase the step size
        pertSize *= 1.2
    print('bestTheta', bestTheta)
    # print(adsa)

    # if not success:
    #   import pdb
    #   pdb.set_trace()

    newTheta = recompThetaSig(prevTheta, bestTheta)
    #print(newTheta)
    newVariance = self.estimVariance(data, dpsCross, clustProbColNormB, newTheta, nrSubjLong)

    return newTheta, newVariance

  def objFunTheta(self, theta, data, dpsCross, clustProbB):

    # print(data.shape, dpsCross.shape)

    sqErrorsSB = np.power((data - self.trajFunc(dpsCross, theta)[:, None]),2)
    meanSSD = np.nansum(np.multiply(clustProbB[None,:], sqErrorsSB), (0,1))
    #print("meanSSD", meanSSD, "clustProbB", clustProbB, "sqErrorsSB", sqErrorsSB)
    #print("ssd/nrSubj", meanSSD/data.shape[0])
    #print(asdsa)

    logPriorTheta = self.logPriorThetaFunc(theta, self.paramsPriorTheta)

    return meanSSD - logPriorTheta, meanSSD

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

    # when there are NaNs, the normalisation of the variance is simply the sum of all the weights
    # corresponding to non-NaN entries
    clustProbSBtiled = np.tile(clustProbB, (crossData.shape[0], 1))
    assert clustProbSBtiled.shape[0] == crossData.shape[0]
    assert clustProbSBtiled.shape[1] == crossData.shape[1]
    weightSumNonNan = np.sum(clustProbSBtiled[np.logical_not(np.isnan(crossData))])

    variance = finalSSD / weightSumNonNan  # variance of biomarker measurement

    return variance

  def loadParamsFromFile(self, paramsDataFile, nrOuterIter, nrInnerIter):
    dataStruct = pickle.load(open(paramsDataFile, 'rb'))
    clustProbBC = dataStruct['clustProbBC']
    thetas = dataStruct['thetas']
    variances = dataStruct['variances']
    subShiftsLong = dataStruct['subShiftsLong']

    thetas = thetas[nrOuterIter - 1, nrInnerIter - 1, :, :]
    variances = variances[nrOuterIter - 1, nrInnerIter - 1, :]
    subShifts = subShiftsLong[nrOuterIter - 1, nrInnerIter - 1, :, :]
    clustProb = clustProbBC

    print('thetas', thetas)
    print('clustProb', clustProb[:,3])

    # place CSF biomk much earlier
    thetas[[1, 9,10,11], 2] = thetas[[1, 9,10,11], 2] - 10

    # place MRI biomk slightly earlier
    thetas[[4,5,6,7,8], 2] = thetas[[4,5,6,7,8], 2] - 5
    print('thetas', thetas)
    print('subShifts.shape', subShifts.shape)
    # print(adsa)


    paramsDataFileNew = '%s/params_2ndFit.npz' % self.outFolder

    return thetas, variances, subShifts, clustProbBC, paramsDataFileNew