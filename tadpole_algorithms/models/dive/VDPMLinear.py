
import voxelDPM
import numpy as np
import scipy
import DisProgBuilder

class VDPMLinearBuilder(DisProgBuilder.DPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, isClust):
    super().__init__(isClust)

  def generate(self, dataIndices, expName, params):
    return VDPMLinear(dataIndices, expName, params)

class VDPMLinear(voxelDPM.VoxelDPM):
  def __init__(self, dataIndices, expName, params):
    super().__init__(dataIndices, expName, params)

  @staticmethod
  def makeThetasIdentif(thetas, shiftTransform):

    #mu, sigma = shiftTransform[0],shiftTransform[1]

    thetas[:,1] += thetas[:,0]*shiftTransform[0] # b = b + a*mu
    thetas[:,0] *= shiftTransform[1] # a = a*sigma

    return thetas

  def initTrajParams(self, crossData, crossDiag, clustProbBC, crossAgeAtScan, subShiftsLong, uniquePartCodeInverse, crossAge1array, extraRangeFactor):
    nrClust = clustProbBC.shape[1]
    thetas = np.zeros((nrClust, 2), float)
    variances = np.zeros(nrClust, float)
    nrSubjCross = crossData.shape[0]
    for c in range(nrClust):
      weightedDataMean = np.average(crossData, axis = 1, weights = clustProbBC[:, c])
      thetas[c,:], residual,_,_ =  np.linalg.lstsq(crossAge1array, weightedDataMean)
      variances[c] = residual/nrSubjCross

    return thetas, variances

  def estimShifts(self, dataOneSubj, thetas, variances, ageOneSubj1array, clustProb,
    prevSubShift, prevSubShiftAvg, fixSpeed):

    objFuncLambda = lambda shift: self.objFunShift(shift, dataOneSubj, thetas,
      variances, ageOneSubj1array, clustProb)
    objFuncDerivLambda = lambda shift: self.objFunShiftDeriv(shift, dataOneSubj,
      thetas, variances, ageOneSubj1array, clustProb)

    res = scipy.optimize.minimize(objFuncLambda, prevSubShift, method='BFGS',
      jac=objFuncDerivLambda, options={'gtol': 1e-8, 'disp': False})

    newShift = res.x
    return newShift

  def objFunShiftDeriv(self, shift, dataOneSubjTB, thetas, variances, ageOneSubj1array, clustProb):

    dps = np.sum(np.multiply(shift, ageOneSubj1array),1)
    #print(ageOneSubj1array.shape)
    nrClust = clustProb.shape[1]
    #for tp in range(dataOneSubj.shape[0]):
    sumSSDalpha = 0
    sumSSDbeta = 0
    for k in range(nrClust):
      errorsSB = dataOneSubjTB - self.trajFunc(dps, thetas[k, :])[:, None]
      betaSqErrorDerivSB = (2 * errorsSB * -thetas[k, 0])
      alphaSqErrorDerivSB = betaSqErrorDerivSB * ageOneSubj1array[:,0][:, None]

      sumSSDalpha += np.sum(np.sum(alphaSqErrorDerivSB, 0) * clustProb[:,k])/ variances[k]
      sumSSDbeta += np.sum(np.sum(betaSqErrorDerivSB, 0) * clustProb[:, k]) / variances[k]

    logPriorShiftDeriv = self.logPriorShiftFuncDeriv(shift, self.paramsPriorShift)

    return np.array([sumSSDalpha - logPriorShiftDeriv[0], sumSSDbeta - logPriorShiftDeriv[1]])

  def estimThetas(self, data, dpsCross, clustProbB, prevTheta, nrSubjLong):

    objFuncLambda = lambda theta: self.objFunTheta(theta, data, dpsCross, clustProbB)
    objFuncDerivLambda = lambda theta: self.objFunThetaDeriv(theta, data, dpsCross, clustProbB)

    res = scipy.optimize.minimize(objFuncLambda, prevTheta, method='BFGS', jac=objFuncDerivLambda,
                                  options={'gtol': 1e-8, 'disp': False})

    newTheta = res.x
    #print(newTheta)
    newVariance = self.estimVariance(data, dpsCross, clustProbB, newTheta, nrSubjLong)

    return newTheta, newVariance

  def objFunThetaDeriv(self, theta, dataSB, dpsCrossS, clustProbB):

    errorsSB = 2*(dataSB - self.trajFunc(dpsCrossS, theta)[:, None])

    errorsSBslope = errorsSB * -dpsCrossS[:, None]
    errorsSBintercept = errorsSB * -1

    meanSSDslope = np.sum(clustProbB[None,:] * errorsSBslope, (0,1))
    meanSSDintercept = np.sum(clustProbB[None, :] * errorsSBintercept, (0, 1))

    return np.array([meanSSDslope, meanSSDintercept])


  def trajFunc(self, s, theta):
    """
    linear function for trectory with params [a,b]
    f(s|theta = [a,b]) = a*s+b

    :param s: the inputs and can be an array of dim N x 1
    :param theta: parameters as np.array([a b])
    :return: values of the linear function at the inputs s
    """

    return theta[0]*s + theta[1]


class VDPMLinearStaticBuilder(DisProgBuilder.DPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self):
    pass

  def generate(self, dataIndices, expName, params):
    return VDPMLinearStatic(dataIndices, expName, params)

class VDPMLinearStatic(VDPMLinear):
  def __init__(self, dataIndices, expName, params):
    super().__init__(dataIndices, expName, params)


  def recompResponsib(self, data, crossAge1array, thetas, variances,
                      subShiftsCross, trajFunc, prevClustProbBC):
    return prevClustProbBC