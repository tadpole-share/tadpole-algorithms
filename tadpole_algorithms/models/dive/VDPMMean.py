import voxelDPM
import numpy as np
import scipy
import DisProgBuilder
import math
import gc
import sys

class VDPMMeanBuilder(voxelDPM.VoxelDPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMMean(dataIndices, expName, params, self.plotterObj)

class VDPMMean(voxelDPM.VoxelDPM):
  def __init__(self, dataIndices, expName, params, plotterObj):
    super().__init__(dataIndices, expName, params, plotterObj)


  def estimShifts(self, dataOneSubj, thetas, variances, ageOneSubj1array, clustProbBC,
    prevSubShift, prevSubShiftAvg, fixSpeed):

    clustProbBCColNorm = clustProbBC / np.sum(clustProbBC, 0)[None, :]

    nrBiomk, nrClust = clustProbBC.shape
    nrTimepts = dataOneSubj.shape[0]
    dataOneSubjWeightedCT = np.zeros((nrClust, nrTimepts), float)
    for c in range(nrClust):
      dataOneSubjWeightedCT[c,:] = np.sum(clustProbBCColNorm[:, c][None,:]
                                            * dataOneSubj, axis=1)

    # if fixSpeed: # fixes parameter alpha to 1
    #   composeShift = lambda beta: [prevSubShiftAvg[0], beta]
    #   initSubShift = prevSubShift[1]
    #   objFuncLambda = lambda beta: self.objFunShift(composeShift(beta), dataOneSubjWeightedCT, thetas,
    #     variances, ageOneSubj1array, clustProbBC)
    #   prevSubShiftAvgCurr = prevSubShiftAvg[1].reshape(1,-1)
    #
    #   objFuncDeriv = lambda shift: self.objFunShiftDeriv(composeShift(beta), dataOneSubjWeightedCT, thetas,
    #     variances, ageOneSubj1array, clustProbBC)
    #
    # else:
    composeShift = lambda shift: shift
    initSubShift = prevSubShift
    objFuncLambda = lambda shift: self.objFunShift(shift, dataOneSubjWeightedCT, thetas,
      variances, ageOneSubj1array, clustProbBC)
    prevSubShiftAvgCurr = prevSubShiftAvg

    objFuncDeriv = lambda shift: self.objFunShiftDeriv(shift, dataOneSubjWeightedCT, thetas,
      variances, ageOneSubj1array, clustProbBC)

    np.seterr(over='ignore')
    np.seterr(invalid='ignore')

    # res = scipy.optimize.minimize(objFuncLambda, initSubShift, method='Nelder-Mead',
    #                               options={'xatol': 1e-2, 'disp': False})

    res = scipy.optimize.minimize(objFuncLambda, initSubShift, method='BFGS', jac=objFuncDeriv,
      options={'gtol': 1e-8, 'disp': False, 'maxiter': 100})


    bestShift = res.x

    # plot func and deriv over alpha
    # minX = bestShift[1]-10
    # maxX = bestShift[1]+10
    # minX = bestShift[0]/100000
    # maxX = bestShift[0]*10
    # minX = 0.10
    # maxX = 3
    #
    # nrPoints = 1000
    # xs = np.linspace(minX, maxX, nrPoints)
    # from matplotlib import pyplot as pl
    # fig = pl.figure(5, figsize=(8,6))
    # # fXs = [objFuncLambda( [bestShift[0], xs[i]]) for i in range(nrPoints)]
    # # dfdXs = [objFuncDeriv([bestShift[0], xs[i]])[1] for i in range(nrPoints)]
    # fXs = [objFuncLambda( [xs[i], bestShift[1]]) for i in range(nrPoints)]
    # dfdXs = [objFuncDeriv([xs[i], bestShift[1]])[0] for i in range(nrPoints)]
    #
    # pl.subplot(2, 1, 1)
    # pl.plot(xs, fXs, c='k', label='obj func')
    # pl.legend()
    # pl.subplot(2, 1, 2)
    # pl.semilogy(xs, dfdXs, c='r', label='deriv')
    # pl.legend()

    # print('xs' ,xs)
    # print('fXs', fXs)
    # print('dfdXs', dfdXs)
    # print(list(zip(xs, fXs, dfdXs)))

    # fig.show()

    # print(objFuncDeriv(bestShift))
    # import pdb
    # pdb.set_trace()


    nrStartPoints = 2
    nrParams = prevSubShiftAvgCurr.shape[0]
    pertSize = 1
    minSSD = res.fun
    success = False
    ssdAll = []
    resAll = []
    ssdInit = minSSD
    for i in range(nrStartPoints):
      perturbShift = prevSubShiftAvgCurr * (np.ones(nrParams) + pertSize *
        np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
      # res = scipy.optimize.minimize(objFuncLambda, perturbShift, method='Nelder-Mead',
      #   options={'xtol': 1e-8, 'disp': False, 'maxiter': 100})

      resPerturb = scipy.optimize.minimize(objFuncLambda, perturbShift, method='BFGS', jac=objFuncDeriv,
        options={'gtol': 1e-8, 'disp': False, 'maxiter': 100})

      currShift = resPerturb.x
      currSSD = resPerturb.fun
      ssdAll += [currSSD]
      resAll += [resPerturb]
      # print('currSSD', currSSD, objFuncLambda(currShift))
      if currSSD < minSSD:
        # if we found a better solution then we decrease the step size
        minSSD = currSSD
        bestShift = currShift
        pertSize /= 1.2
        success = resPerturb.success
      else:
        # if we didn't find a solution then we increase the step size
        pertSize *= 1.2
    # print('bestShift', bestShift)

    # print('ssdAll', ssdAll)

    # import pdb
    # pdb.set_trace()

    np.seterr(over='warn')
    np.seterr(invalid='warn')

    return composeShift(bestShift)

  def objFunShift(self, shift, dataOneSubjWeightedCT, thetas, variances,
                  ageOneSubj1array, clustProbBC):

    # print('dataOneSubjWeightedCT', dataOneSubjWeightedCT.dtype)
    # print('ageOneSubj1array', ageOneSubj1array.dtype)
    # print('clustProbBC', clustProbBC.dtype)
    # print(adsas)

    dps = np.sum(np.multiply(shift, ageOneSubj1array), 1)
    nrClust = thetas.shape[0]
    # for tp in range(dataOneSubj.shape[0]):
    sumSSD = 0
    gammaInvK = np.sum(clustProbBC, 0)
    for k in range(nrClust):
      sqError = np.sum(np.power(dataOneSubjWeightedCT[k,:] -
                                  self.trajFunc(dps, thetas[k, :]), 2))
      sumSSD += (sqError * gammaInvK[k])/ (2 * variances[k])

    logPriorShift = self.logPriorShiftFunc(shift, self.paramsPriorShift)

    # print('logPriorShift', logPriorShift, 'sumSSD', sumSSD)
    # print(sumSSD)
    # if shift[0] < -400: # and -67
    #   import pdb
    #   pdb.set_trace()
    if not (np.isfinite(sumSSD) and np.isfinite(logPriorShift)) and shift[0] > 0:
      import pdb
      pdb.set_trace()


    return sumSSD - logPriorShift


  def objFunShiftDeriv(self, shift, dataOneSubjWeightedCT, thetas, variances, ageOneSubj1array, clustProbBC):

    aK = thetas[:, 0]
    bK = thetas[:, 1]
    cK = thetas[:, 2]

    dpsT = np.sum(shift * ageOneSubj1array,1)
    #print(ageOneSubj1array.shape)
    nrClust = clustProbBC.shape[1]
    #for tp in range(dataOneSubj.shape[0]):
    sumSqErrorDerivalpha = 0
    sumSqErrorDerivbeta = 0
    gammaInvK = np.sum(clustProbBC, 0)

    alphaSumDiffs = 0
    betaSumDiffs = 0

    for k in range(nrClust):

      # sqError = np.sum(np.power(dataOneSubjWeightedCT[k,:] -
      #                             self.trajFunc(dps, thetas[k, :]), 2))
      # sumSSD += (sqError * gammaInvK[k])/ (2 * variances[k])

      # dQ/d alpha_i = \sum(k=1)^K 1/(2 sigma^2) \sum_{j} *
      # 2(dataOneSubjWeightedCT_k - f(alpha_ t_ij + beta_i|theta_k)) -df(.)/d alpha_i

      scalingTerm = gammaInvK[k]/ variances[k]
      errorsT = scalingTerm * (dataOneSubjWeightedCT[k,:] - self.trajFunc(dpsT, thetas[k, :]))

      # exp(-bk(alpha_i * t_ij + beta_i - ck))
      expTerm = np.exp(-bK[k]*(dpsT-cK[k]))

      # ak*(1+exp(-bk(alpha_i * t_ij + beta_i - ck)))^-2
      derivTerm1T = aK[k]*np.power((1+expTerm),-2)

      # exp(-bk(alpha_i * t_ij + beta_i - ck))
      derivTerm2T = derivTerm1T * expTerm

      # if overflow set derivTerm to zero, which is mathematically correct
      derivTerm2T[np.isinf(expTerm)] = 0

      # *= -bk * t_ij
      alphaDerivTermT = derivTerm2T * (-bK[k] * ageOneSubj1array[:,0])
      # *= -bk
      betaDerivTermT = derivTerm2T * -bK[k]

      alphaSumDiffs += np.sum(errorsT * alphaDerivTermT)
      betaSumDiffs += np.sum(errorsT * betaDerivTermT)

      # if np.isinf(expTerm).any():
      #   import pdb
      #   pdb.set_trace()

      if np.isnan(alphaSumDiffs):
        import pdb
        pdb.set_trace()


    logPriorShiftDeriv = self.logPriorShiftFuncDeriv(shift, self.paramsPriorShift)


    # if alphaSumDiffs == 0:
    #   pdb.set_trace()

    return np.array([alphaSumDiffs - logPriorShiftDeriv[0], betaSumDiffs - logPriorShiftDeriv[1]])

  # def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong):
  #
  #   recompThetaSig = lambda thetaFull, theta12: [thetaFull[0], theta12[0], theta12[1], thetaFull[3]]
  #
  #   dataWeightedS = np.sum(np.multiply(clustProbB[None, :], data), axis=1)
  #   objFuncLambda = lambda theta12: self.objFunTheta(recompThetaSig(prevTheta, theta12),
  #     dataWeightedS, dpsCross, clustProbB)[0]
  #
  #   # objFuncDerivLambda = lambda theta: self.objFunThetaDeriv(theta, data, dpsCross, clustProbB)
  #
  #   # res = scipy.optimize.minimize(objFuncLambda, prevTheta, method='BFGS', jac=objFuncDerivLambda,
  #   #                               options={'gtol': 1e-8, 'disp': False})
  #
  #   initTheta12 = prevTheta[[1, 2]]
  #   res = scipy.optimize.minimize(objFuncLambda, initTheta12, method='Nelder-Mead',
  #                                 options={'xtol': 1e-8, 'disp': True})
  #
  #   newTheta = recompThetaSig(prevTheta, res.x)
  #   #print(newTheta)
  #   newVariance = self.estimVariance(data, dpsCross, clustProbB, newTheta, nrSubjLong)
  #
  #   return newTheta, newVariance

  def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong, prevThetas):

    dataWeightedS = np.sum(np.multiply(clustProbB[None, :], data), axis=1)
    objFuncLambda = lambda theta: self.objFunTheta(theta, dataWeightedS, dpsCross, clustProbB)[0]

    initTheta = prevTheta

    nrStartPoints = 15
    nrParams = initTheta.shape[0]
    pertSize = 1
    minTheta = np.array([-np.inf, -1/np.std(dpsCross), -np.inf, -np.inf])
    maxTheta = np.array([np.inf, 0, np.inf, np.inf])
    varParams = (0.2 * np.std(prevThetas, axis=0))**2

    res = scipy.optimize.minimize(objFuncLambda, initTheta, method='Nelder-Mead',
      options={'xtol': 1e-4, 'disp': True})
    minSSD = res.fun
    bestTheta = res.x

    # success = False
    for i in range(nrStartPoints):
      perturbTheta = initTheta * (np.ones(nrParams) + pertSize *
        np.random.multivariate_normal(np.zeros(nrParams), np.diag(varParams)))
      # print('perturbTheta < minTheta', perturbTheta < minTheta)
      # perturbTheta[perturbTheta < minTheta] = minTheta[perturbTheta < minTheta]
      # perturbTheta[perturbTheta > maxTheta] = minTheta[perturbTheta > maxTheta]
      res = scipy.optimize.minimize(objFuncLambda, perturbTheta, method='Nelder-Mead',
        options={'xtol': 1e-4, 'disp': True})
      currTheta = res.x
      currSSD = res.fun
      # print('currSSD', currSSD, objFuncLambda(currTheta))
      if currSSD < minSSD:
        # if we found a better solution then we decrease the step size
        minSSD = currSSD
        bestTheta = currTheta
        pertSize /= 1.2
        # success = res.success
      else:
        # if we didn't find a solution then we increase the step size
        pertSize *= 1.2

    newTheta = bestTheta
    newVariance = self.estimVariance(data, dpsCross, clustProbB, newTheta, nrSubjLong)

    return newTheta, newVariance

  # def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong):
  #
  #   recompThetaSig = lambda thetaFull, theta12: [thetaFull[0], theta12[0], theta12[1], thetaFull[3]]
  #
  #   dataWeightedS = np.sum(np.multiply(clustProbB[None, :], data), axis=1)
  #   objFuncLambda = lambda theta12: self.objFunTheta(recompThetaSig(prevTheta, theta12),
  #     dataWeightedS, dpsCross, clustProbB)[0]
  #
  #   # objFuncDerivLambda = lambda theta: self.objFunThetaDeriv(theta, data, dpsCross, clustProbB)
  #
  #   # res = scipy.optimize.minimize(objFuncLambda, prevTheta, method='BFGS', jac=objFuncDerivLambda,
  #   #                               options={'gtol': 1e-8, 'disp': False})
  #
  #   initTheta12 = prevTheta[[1, 2]]
  #
  #   nrStartPoints = 10
  #   nrParams = initTheta12.shape[0]
  #   pertSize = 1
  #   minTheta = np.array([-1/np.std(dpsCross), -np.inf])
  #   maxTheta = np.array([0, np.inf])
  #   minSSD = np.inf
  #   bestTheta = initTheta12
  #   success = False
  #   for i in range(nrStartPoints):
  #     perturbTheta = initTheta12 * (np.ones(nrParams) + pertSize *
  #       np.random.multivariate_normal(np.zeros(nrParams), np.eye(nrParams)))
  #     # print('perturbTheta < minTheta', perturbTheta < minTheta)
  #     # perturbTheta[perturbTheta < minTheta] = minTheta[perturbTheta < minTheta]
  #     # perturbTheta[perturbTheta > maxTheta] = minTheta[perturbTheta > maxTheta]
  #     res = scipy.optimize.minimize(objFuncLambda, perturbTheta, method='Nelder-Mead',
  #       options={'xtol': 1e-8, 'disp': True, 'maxiter':100})
  #     currTheta = res.x
  #     currSSD = res.fun
  #     # print('currSSD', currSSD, objFuncLambda(currTheta))
  #     if currSSD < minSSD:
  #       # if we found a better solution then we decrease the step size
  #       minSSD = currSSD
  #       bestTheta = currTheta
  #       pertSize /= 1.2
  #       success = res.success
  #     else:
  #       # if we didn't find a solution then we increase the step size
  #       pertSize *= 1.2
  #   # print('bestTheta', bestTheta)
  #   # print(adsa)
  #
  #   # if not success:
  #   #   import pdb
  #   #   pdb.set_trace()
  #
  #   newTheta = recompThetaSig(prevTheta, bestTheta)
  #   #print(newTheta)
  #   newVariance = self.estimVariance(data, dpsCross, clustProbB, newTheta, nrSubjLong)
  #
  #   return newTheta, newVariance


  def objFunTheta(self, theta, dataWeightedS, dpsCross, _):

    sqErrorsS = np.power((dataWeightedS - self.trajFunc(dpsCross, theta)), 2)
    meanSSD = np.sum(sqErrorsS)

    logPriorTheta = self.logPriorThetaFunc(theta, self.paramsPriorTheta)

    return meanSSD - logPriorTheta, meanSSD

  def estimVariance(self, crossData, dpsCross, clustProbB, theta, nrSubjLong):
    dataWeightedS = np.sum(np.multiply(clustProbB[None, :], crossData), axis=1)
    finalSSD = self.objFunTheta(theta, dataWeightedS, dpsCross, clustProbB)[1]
    # remove the degrees of freedom: 2 for each subj (slope and shift) and one for each parameters in the model
    # variance = finalSSD / (crossData.shape[0] -2*nrSubjLong - theta.shape[0])  # variance of biomarker measurement
    variance = finalSSD / (crossData.shape[0])  # variance of biomarker measurement

    return variance


  # def objFunThetaDeriv(self, theta, dataSB, dpsCrossS, clustProbB):
  #
  #   errorsSB = 2*(dataSB - self.trajFunc(dpsCrossS, theta)[:, None])
  #
  #   errorsSBslope = errorsSB * -dpsCrossS[:, None]
  #   errorsSBintercept = errorsSB * -1
  #
  #   meanSSDslope = np.sum(clustProbB[None,:] * errorsSBslope, (0,1))
  #   meanSSDintercept = np.sum(clustProbB[None, :] * errorsSBintercept, (0, 1))
  #
  #   return np.array([meanSSDslope, meanSSDintercept])

  def recompResponsib(self, crossData, longData, crossAge1array, thetas, variances, subShiftsCross,
    trajFunc, prevClustProbBC, scanTimepts, partCode, uniquePartCode, outerIt):
    # overwrite function as we need to use a different variance (in the biomk measurements as opposed to their mean)

    prevClustProbColNormBC = prevClustProbBC / np.sum(prevClustProbBC, 0)[None, :]
    (nrSubj, nrBiomk) = crossData.shape
    nrClust = thetas.shape[0]
    dps = voxelDPM.VoxelDPM.calcDps(subShiftsCross, crossAge1array)
    varianceIndivBiomk = np.zeros(variances.shape, float)
    # estimate the variance in the biomk noise, as opposed to the variance in the mean
    for c in range(nrClust):
      # call super method
      finalSSD = super(VDPMMean,self).objFunTheta(thetas[c,:], crossData, dps,
        prevClustProbColNormBC[:,c])[1]
      varianceIndivBiomk[c] = finalSSD / (crossData.shape[0])

    fSK = np.zeros((nrSubj, nrClust), float)
    for k in range(nrClust):
      fSK[:, k] = trajFunc(dps, thetas[k, :])

    logClustProb = np.zeros((nrBiomk, nrClust), float)
    clustProb = np.zeros((nrBiomk, nrClust), float)
    tmpSSD = np.zeros((nrBiomk, nrClust), float)
    tmpSSDVar = np.zeros((nrBiomk, nrClust), float)
    for k in range(nrClust):
      tmpSSD[:, k] = np.sum(np.power(crossData - fSK[:, k][:, None], 2),
                            0)  # sum across subjects, left with 1 x NR_BIOMK array
      assert (tmpSSD[:, k].shape[0] == nrBiomk)
      tmpSSDVar[:, k] = -tmpSSD[:, k] / (2 * varianceIndivBiomk[k])
      logClustProb[:, k] = -tmpSSD[:, k] / (2 * varianceIndivBiomk[k]) - np.log(2 * math.pi * varianceIndivBiomk[k]) * nrSubj / 2

    vertexNr = 755
    # print('tmpSSD[vertexNr,:]', tmpSSD[vertexNr, :])  # good
    # print('tmpSSDVar[vertexNr,:] ', tmpSSDVar[vertexNr, :])  # good
    # print('logClustProb[vertexNr,:]', logClustProb[vertexNr, :])  # bad

    for k in range(nrClust):
      expDiffs = np.power(np.e, logClustProb - logClustProb[:, k][:, None])
      clustProb[:, k] = np.divide(1, np.sum(expDiffs, axis=1))

    # for c in range(nrClust):
    #   print('sum%d' % c, np.sum(clustProb[:, c]))

    # import pdb
    # pdb.set_trace()

    return clustProb, crossData, longData, np.nan


  def calcModelLogLik(self, data, dpsCross, thetas, variances, clustProbBC):
    """ computed the full model log likelihood, used for checking if it increases during EM and for BIC"""

    nrBiomk = data.shape[1]
    nrClust = thetas.shape[0]

    prodLikBC = np.longdouble(np.zeros((nrBiomk, nrClust), float))

    prevClustProbColNormBC = clustProbBC / np.sum(clustProbBC, 0)[None, :]
    varianceIndivBiomk = np.zeros(variances.shape, float)
    # estimate the variance in the biomk noise, as opposed to the variance in the mean
    for c in range(nrClust):
      # call super method
      finalSSD = super(VDPMMean,self).objFunTheta(thetas[c,:], data, dpsCross,
        prevClustProbColNormBC[:,c])[1]
      varianceIndivBiomk[c] = finalSSD / (data.shape[0])

    for c in range(nrClust):
      sqErrorsSB = np.power((data - self.trajFunc(dpsCross, thetas[c, :])[:, None]), 2)  # taken from estimTheta
      pdfSB = (2 * math.pi * varianceIndivBiomk[c]) ** (-1/2) * np.exp(-(2 * varianceIndivBiomk[c]) ** (-1) * sqErrorsSB)

      # np.prod(pdfSB, axis=0)
      prodLikBC[:, c] = (1 / nrClust) * np.prod(np.longdouble(pdfSB), axis = 0) # it is product here as the log doesn't go this far

      # prodLikBC[b,c] = (1/nrClust) * np.prod(scipy.stats.norm(data[:,b], loc=self.trajFunc(dpsCross, thetas[c,:]),
      #   scale=variances[c]))

    logLik = np.sum(np.log(np.sum(prodLikBC, axis = 1)))

    # print('logLik', logLik)
    # import pdb
    # pdb.set_trace()

    return logLik


  def calcModelLogLikFromEnergy(self, data, dpsCross, thetas, variances, clustProbBC, nrSubjLong):
    """ computed the full model log likelihood from the Energy (used in EM) and Entropy over Z"""

    nrBiomk = data.shape[1]
    nrClust = thetas.shape[0]

    prevClustProbColNormBC = clustProbBC / np.sum(clustProbBC, 0)[None, :]
    varianceIndivBiomk = np.zeros(variances.shape, float)
    # estimate the variance in the biomk noise, as opposed to the variance in the mean
    for c in range(nrClust):
      # call super method
      finalSSD = super(VDPMMean,self).objFunTheta(thetas[c,:], data, dpsCross,
        prevClustProbColNormBC[:,c])[1]
      varianceIndivBiomk[c] = finalSSD / (data.shape[0])

    prevClustProbColNormBC = None
    gc.collect()
    print('prevClustProb gc collected')
    sys.stdout.flush()

    # first calculate the energy term used in EM
    sumLogLikC = np.zeros((nrClust), np.float32)
    for c in range(nrClust):
      # dataWeightedS = np.sum(np.multiply(clustProbBC[:,c][None, :], data), axis = 1)

      sqErrorsSB = np.power((data - self.trajFunc(dpsCross, thetas[c, :])[:, None]), 2)  # taken from estimTheta
      logpdfSB = -np.log(2 * math.pi * varianceIndivBiomk[c]) / 2 - (2 * varianceIndivBiomk[c]) ** (-1) * sqErrorsSB

      # np.prod(pdfSB, axis=0)
      sumLogLikC[c] = np.sum(clustProbBC[:, c][None, :] * np.sum(logpdfSB, axis = 0), axis = (0,1))

      # prodLikBC[b,c] = (1/nrClust) * np.prod(scipy.stats.norm(data[:,b], loc=self.trajFunc(dpsCross, thetas[c,:]),
      #   scale=variances[c]))

    logLikEnergy = np.sum(sumLogLikC, axis = 0)

    # calculate the entropy term
    logClustProbBC = np.nan_to_num(np.log(clustProbBC))
    # logClustProbBC[np.isnan(logClustProbBC)] = 0
    logLikEntropy = -np.sum(clustProbBC * logClustProbBC, axis = (0, 1))

    logLik = logLikEnergy + logLikEntropy

    # print('logLikEnergy', logLikEnergy)
    # print('logLikEntropy', logLikEntropy)
    # print('logLik', logLik)

    # import pdb
    # pdb.set_trace()

    # calculate BIC and AIC
    nrDataPoints = data.shape[0] * data.shape[1]

    # should I include the clustering prob in the nr Free params? perhaps, as they are optimised in E-step
    nrFreeParams = nrClust * (thetas.shape[1] + 1) + 2 * nrSubjLong
    nrFreeParams = nrFreeParams + clustProbBC.shape[0] * (clustProbBC.shape[1]-1)

    bic = -2 * logLik + nrFreeParams * np.log(nrDataPoints)
    aic = -2 * logLik + nrFreeParams * 2

    # print('bic')

    return logLik, bic, aic
