import numpy as np
import nibabel as nib
import colorsys
import scipy.stats
import sys
import pickle
import os
import sklearn.cluster
import copy

from env import *
import evaluationFramework

# def biasCorrection(data, diag, regressorVector):
#
#   regressorCTL = regressorVector[diag == CTL]
#
#   X2 = np.concatenate((regressorCTL.reshape(-1,1), np.ones(regressorCTL.reshape(-1,1).shape)), axis=1)
#   # Solve the GLM: Y = [X 1] * M
#   XXX_2X = np.dot(np.pinv(np.dot(X2.T, X2)), X2.T)
#   M_2B = np.dot(XXX_2X, data[diag == CTL, :])  # params of   linear   fit
#   RegressorCTL1array_X2 = np.concatenate((regressorCTL.reshape(-1,1), np.ones(regressorCTL.reshape(-1,1).shape)), axis=1)
#   Yhat_XB = np.dot(RegressorCTL1array_X2, M_2B)
#   assert (~any(np.isnan(M_2B)));
#   newData = data - (Yhat_XB - np.mean(data[diag == CTL, :], axis=0)[None,:])
#
#   for i in range(data.shape[1]):


def makeAvgBiomkMaps(data, diag, ageAtScan, plotTrajParams, datasetLabel, fwhmLevel,
                         diagLabels,diagIndices=None):

  fsaverageThickFile = '%s/subjects/fsaverage/surf/lh.thickness' % plotTrajParams['freesurfPath']
  fsAvgDataStruct = nib.freesurfer.io.read_morph_data(fsaverageThickFile)
  print(fsAvgDataStruct.shape)

  if diagIndices is None:
    diagIndices = list(np.unique(diag))

  nrUnqDiags = len(diagIndices)



  for d in range(nrUnqDiags):
    thickFile = 'resfiles/%savgMapFWHM%d%s' % (datasetLabel, fwhmLevel,
      diagLabels[diagIndices[d]])
    avgThickCurrDiag = np.zeros(fsAvgDataStruct.shape)
    avgThickCurrDiag[plotTrajParams['pointIndices']] = -np.mean(data[diag == diagIndices[d],:], axis=0)
    print(diag, diagIndices[d])
    print(avgThickCurrDiag[::5000])
    print('thickFile %s' % thickFile)
    print('nr with diag %d: ' % diagIndices[d], np.sum(diag == diagIndices[d]))
    nib.freesurfer.io.write_morph_data(thickFile, avgThickCurrDiag)


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

    assert(len(currPartTimeptsOrdInd) >= 2) # 2 for PET, 3 for MRI

    if len(currPartTimeptsOrdInd) > 1:
      longArray += [array[currPartIndicesOrd]]

  return longArray

def createLongData(data, diag, scanTimepts, partCode, ageAtScan):

  uniquePartCode = np.unique(partCode)

  longData = makeLongArray(data, scanTimepts, partCode, uniquePartCode)
  longDiagAllTmpts = makeLongArray(diag, scanTimepts, partCode, uniquePartCode)
  longDiag = np.array([x[0] for x in longDiagAllTmpts])
  longScanTimepts = makeLongArray(scanTimepts, scanTimepts, partCode, uniquePartCode)
  longPartCodeAllTimepts = makeLongArray(partCode, scanTimepts, partCode, uniquePartCode)
  longPartCode = np.array([x[0] for x in longPartCodeAllTimepts])
  longAgeAtScan = makeLongArray(ageAtScan, scanTimepts, partCode, uniquePartCode)
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

  return longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan, inverseMap, filtData, filtDiag, filtScanTimetps, filtPartCode, filtAgeAtScan


def orderTrajBySlope(thetas):
  slopes = (thetas[:, 0] * thetas[:, 1]) / 4
  clustOrderInd = np.argsort(slopes)  # -0.3 -0.7 -1.2   green -> yelllow -> red
  # print('slopes, clustOrderInd', slopes, clustOrderInd)
  return clustOrderInd

def orderClustByAtrophyExtent(thetas, trajFunc):
  nrClust = thetas.shape[0]
  dpsThresh = 3
  print('dpsThresh', dpsThresh)
  biomkValuesThresh = np.zeros(nrClust, float)
  for c in range(nrClust):
    biomkValuesThresh[c] = trajFunc(dpsThresh, thetas[c, :])

  sortedInd = np.argsort(biomkValuesThresh)

  return sortedInd, biomkValuesThresh[sortedInd]


def colorClustBySlope(thetas):
  ''' Generates colors for the clusters! WARNING: the colors are not given in the cluster order as defined by thetas'''
  minHue = 0
  maxHue = 0.66
  slopes = (thetas[:, 0] * thetas[:, 1]) / 4
  nrClust = thetas.shape[0]
  slopesSortedInd = np.argsort(slopes)
  clustColsRGB = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(minHue, maxHue, nrClust, endpoint=True)]

  clustColsRGBperturb = [colorsys.hsv_to_rgb(hue, 0.3, 1) for hue in
    np.linspace(minHue, maxHue, nrClust, endpoint=True)]

  return clustColsRGB, clustColsRGBperturb, slopesSortedInd

def makeClustProbFromArray(initClust):
  # clusters are numbered from 0 to K-1 in initCLust

  assert(np.min(initClust) == 0)
  nrBiomk = initClust.shape[0]
  nrClust = np.max(initClust)+1

  clustProbBC = np.zeros((nrBiomk, nrClust), float)
  for b in range(nrBiomk):
    # print("clustProbOBC", b, initClust[b]-1, nrBiomk)
    clustProbBC[b, initClust[b]] = 1

  clustProbBC += 0.001 # add some prob just in case some clusters have no points assigned
  clustProbBC = clustProbBC / np.sum(clustProbBC,axis=1)[:, None]

  # print(clustProbBC[0,:])

  return clustProbBC

def logPriorShiftUnif(shift, paramsPriorShift):
  """uniform prior"""
  return 0

def logPriorShiftUnifDeriv(shift, paramsPriorShift):
  """uniform prior"""
  return np.zeros(shift.shape[0])

def logPriorShiftInform(shift, paramsPriorShift):
  """returns the log prior on shifts p(alpha_i, beta_i) such that:
   log p(alpha_i, beta_i) = log p(alpha_i) + log p(beta_i)

   p(alpha_i) ~ Gamma(shape, rate)
   p(beta_i) ~ Gaussian(mu, std)

   paramsPriorShift = [shape, rate, mu, std]

   """
  shape = paramsPriorShift[0]
  rate = paramsPriorShift[1]
  mu = paramsPriorShift[2]
  std = paramsPriorShift[3]

  alpha = shift[0]
  beta = shift[1]

  logLikAlpha = scipy.stats.gamma.logpdf(alpha, a=shape, scale=1.0/rate)
  logLikBeta = scipy.stats.norm.logpdf(beta, loc=mu, scale=std)

  # if logLikAlpha == 0 or logLikBeta == 0:
  #   return -float('inf')

  return logLikAlpha + logLikBeta

def logPriorShiftInformDeriv(shift, paramsPriorShift):
  """returns the derivative of the log prior on shifts p(alpha_i, beta_i) such that:
   d/d_a log p(alpha_i, beta_i) =d/d_a log p(alpha_i) + d/d_a log p(beta_i)

   p(alpha_i) ~ Gamma(shape, rate)
   p(beta_i) ~ Gaussian(mu, std)

   paramsPriorShift = [shape, rate, mu, std]

   """
  shape = paramsPriorShift[0]
  rate = paramsPriorShift[1]
  mu = paramsPriorShift[2]
  std = paramsPriorShift[3]

  alpha = shift[0]
  beta = shift[1]

  logpAlpha = (shape-1)/alpha - rate

  logpBeta = (mu - beta)/(std**2)

  return np.array([logpAlpha, logpBeta])

def logPriorThetaUnif(theta, paramsPriorTheta):
  """uniform prior"""
  return 0

def logPriorThetaUnifDeriv(theta, paramsPriorTheta):
  """uniform prior"""
  return np.zeros(theta.shape[0])

def logPriorThetaInform(theta, paramsPriorTheta):
  """returns the log prior on sigmoid parameters p(a, b, c, d| paramsPriorTheta) such that:
   log p(a, b, c, d|.) = log (a|.) + log (b|.) + log (c|.) + log (d|.)

   p(a) ~ N(mu_a, sigma_a)
   p(b) ~ N(mu_b, sigma_b)
   ...

   paramsPriorShift = [shape, rate, mu, std]

   """
  mu_a = paramsPriorTheta[0]
  std_a = paramsPriorTheta[1]
  mu_b = paramsPriorTheta[2]
  std_b = paramsPriorTheta[3]
  mu_c = paramsPriorTheta[4]
  std_c = paramsPriorTheta[5]
  mu_d = paramsPriorTheta[6]
  std_d = paramsPriorTheta[7]

  a = theta[0]
  b = theta[1]
  c = theta[2]
  d = theta[3]

  likA = scipy.stats.norm.pdf(a, loc=mu_a, scale=std_a)
  likB = scipy.stats.norm.pdf(b, loc=mu_b, scale=std_b)
  likC = scipy.stats.norm.pdf(c, loc=mu_c, scale=std_c)
  likD = scipy.stats.norm.pdf(d, loc=mu_d, scale=std_d)


  if likA == 0 or likB == 0 or likC == 0 or likD == 0:
    # print('paramsPriorTheta', paramsPriorTheta)
    # print('mu_a', mu_a, 'std_a', std_a, 'mu_d', mu_d, 'std_d', std_d)
    # print('theta ', theta, 'a ', a, 'b ', b, 'c ', c, 'd ', d)
    # print(likA, likB, likC, likD)
    return -float('inf')

  return np.log(likA) + np.log(likB) + np.log(likC) + np.log(likD)

def logPriorThetaInformDeriv(theta, paramsPriorTheta):
  """returns the derivative of the log prior on shifts p(alpha_i, beta_i) such that:
   d/d_a log p(alpha_i, beta_i) =d/d_a log p(alpha_i) + d/d_a log p(beta_i)

   p(alpha_i) ~ Gamma(shape, rate)
   p(beta_i) ~ Gaussian(mu, std)

   paramsPriorShift = [shape, rate, mu, std]

   """
  raise ValueError('Prior derivative on theta not implemented yet')


def sigmoidFunc(s, theta):
  """
  sigmoidal function for trectory with params [a,b,c,d] with
  minimum d, maximum a+d, slope a*b/4 and slope
  maximum attained at center c
  f(s|theta = [a,b,c,d]) = a/(1+exp(-b(s-c)))+d

  :param s: the inputs and can be an array of dim N x 1
  :param theta: parameters as np.array([a b c d])
  :return: values of the sigmoid function at the inputs s
  """

  return theta[0] * np.power((1 + np.exp(-theta[1] * (s - theta[2]))), -1) + theta[3]


def linearFunc(s, theta):
    """
    linear function for trectory with params [a,b]
    f(s|theta = [a,b]) = a*s+b

    :param s: the inputs and can be an array of dim N x 1
    :param theta: parameters as np.array([a b])
    :return: values of the linear function at the inputs s
    """

    return theta[0]*s + theta[1]

def printStats(longData, longDiag):
  avgNrScans = np.mean([longData[i].shape[0] for i in range(len(longData))])
  nrSubjCross = len(longData)
  print(avgNrScans, nrSubjCross)

  print('nrCTL', np.sum(longDiag == CTL))
  print('nrPCA', np.sum(longDiag == PCA))
  print('nrAD', np.sum(longDiag == AD))



  import pdb
  pdb.set_trace()


def findNearNeigh(runPart, datasetFull, pointIndices, freesurfPath, indSortedAbnorm):
  nnDataFile = 'resfiles/%sNearNeigh.npz' % datasetFull

  # need to do this even for the entire dataset, as a very small # of vertices are removed
  # this is due to errors in FS alignment.
  sys.stdout.flush()
  if runPart == 'R':

    print(pointIndices.shape)
    fsaverageInflatedLh = '%s/subjects/fsaverage/surf/lh.inflated' % \
                          freesurfPath
    coordsLh, facesLh, _ = nib.freesurfer.io.read_geometry(fsaverageInflatedLh, read_metadata=True)

    V = coordsLh.shape[0]  # nr of total vertices in fsaverage
    P = pointIndices.shape[0]  # number of selected vertices out of V (slightly smaller)
    # nearestNeighbours = np.array(range(V))
    nearestNeighboursFast = np.zeros(V, int)

    chunkLen = 500
    nrChunks = int(V / chunkLen)  # each chunk has around 1000 vertices

    coordsLh.astype(np.float16)  # cast to smaller precision to save space for chunking.
    pointCoords = coordsLh[pointIndices, :]

    nearestNeighboursFast[pointIndices] = range(len(pointIndices))
    leftoverPointIndices = np.setdiff1d(np.array(range(V)), pointIndices, assume_unique=True)
    print(leftoverPointIndices)
    # print(adsa)

    for v in range(leftoverPointIndices.shape[0]):
      nearestNeighboursFast[leftoverPointIndices[v]] = np.argmin(
        np.linalg.norm(coordsLh[leftoverPointIndices[v], :][None, :]
                       - coordsLh[pointIndices, :], ord=2, axis=1))

    print(np.max(nearestNeighboursFast), P)
    assert(np.max(nearestNeighboursFast) == (P - 1))
    nearestNeighbours = nearestNeighboursFast

    print('facesLh', facesLh)

    adjListHelper = np.zeros(P, dtype=object)
    for p in range(P):
      adjListHelper[p] = []
    nrFaces = facesLh.shape[0]
    for f in range(nrFaces):
      pointsFaces = nearestNeighboursFast[facesLh[f, :]]
      adjListHelper[pointsFaces[0]] += [pointsFaces[1], pointsFaces[2]]
      adjListHelper[pointsFaces[1]] += [pointsFaces[0], pointsFaces[2]]
      adjListHelper[pointsFaces[2]] += [pointsFaces[0], pointsFaces[1]]

    # assumes freesurfer structure
    # for l in [4, 5, 6, 7, 8]:
    #   lenList = [1 for p in range(P) if np.unique(adjListHelper[p]).shape[0] == l]
    #   print('lenList %d' % l, np.sum(lenList))

    P = pointIndices.shape[0]
    targetSize = 6
    adjList = np.zeros((P, targetSize), int)
    actualSize = np.zeros(P, int)
    for p in range(P):
      # print(p, adjListHelper[p], np.unique(adjListHelper[p]))
      adjListCurr = np.unique(adjListHelper[p])
      actualSize[p] = adjListCurr.shape[0]
      if adjListCurr.shape[0] > targetSize:
        # if too many neighbours, just take the first targetSize
        adjList[p, :] = adjListCurr[:targetSize]
      elif adjListCurr.shape[0] < targetSize:
        # if less than targetSize neighbours, just repeat a few
        adjList[p, :adjListCurr.shape[0]] = adjListCurr
        adjList[p, adjListCurr.shape[0]:] = adjList[p, :(targetSize - adjListCurr.shape[0])]
      else:
        adjList[p, :] = adjListCurr

    print('1st deg - mean actual size', np.mean(actualSize))
    print('adjList[0]', adjList[0], adjList[adjList[0]], np.unique(adjList[adjList[0]].reshape(-1)))

    # try 2nd degree Markov chain, i.e. each vertex will have its immediate neighbours
    # and the neighbours of neighbours. Can extend to 3rd, 4th degree, etc ..
    targetSize = [0,0,19,36] # 18 should be a good number, assuming each vertex has 6 neighbours on avg.
    deg = 3
    adjListHigherDeg = np.zeros((P, targetSize[deg]), int)
    actualSize = np.zeros(P, int)
    for p in range(P):
      adjListCurr = list(adjList[p,:])
      for d in range(deg-1): # for deg 2 we only need one iteration
        adjListCurr += list(adjList[adjListCurr].reshape(-1))
      adjListCurr = np.unique(adjListCurr)
      actualSize[p] = adjListCurr.shape[0]

      if adjListCurr.shape[0] > targetSize[deg]:
        # if too many neighbours, just take the first targetSize
        adjListHigherDeg[p, :] = adjListCurr[:targetSize[deg]]
      elif adjListCurr.shape[0] < targetSize[deg]:
        # if less than targetSize neighbours, just repeat a few
        adjListHigherDeg[p, :adjListCurr.shape[0]] = adjListCurr
        adjListHigherDeg[p, adjListCurr.shape[0]:] = adjListHigherDeg[p, :(targetSize[deg] - adjListCurr.shape[0])]
      else:
        adjListHigherDeg[p, :] = adjListCurr

    print('2nd deg - mean actual size', np.mean(actualSize))
    print(adjListHigherDeg[:1,:])


    # now find a smaller subset, like 10k, on which to run the k-means clustering
    # assign the same cluster to the other nearby points
    nrPointsToSample = 10000
    initClustSubsetInd = np.random.choice(range(P), size=nrPointsToSample, replace=False)
    kmeansCoords = pointCoords[initClustSubsetInd,:]
    # these will map to the pointIndices, which then map to fsaverage vertices
    nearNeighInitClust = np.zeros(P, int)
    for v in range(P):
      nearNeighInitClust[v] = np.argmin(
        np.linalg.norm(pointCoords[v, :][None, :]
                       - kmeansCoords, ord=2, axis=1))

    print('nearNeighInitClust', nearNeighInitClust)
    print('nearNeighInitClust[initClustSubsetInd]', nearNeighInitClust[initClustSubsetInd][:30])
    print('initClustSubsetInd[:30]', initClustSubsetInd[:30])
    assert (nearNeighInitClust[initClustSubsetInd] == np.array(range(nrPointsToSample))).all()

    nnDataStruct = dict(pointIndices=pointIndices, nearestNeighbours=nearestNeighbours, adjList=adjList
      ,initClustSubsetInd=initClustSubsetInd, nearNeighInitClust=nearNeighInitClust,
      adjListHigherDeg=adjListHigherDeg, indSortedAbnorm=indSortedAbnorm)
    pickle.dump(nnDataStruct, open(nnDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
  elif runPart == 'L':
    nnDataStruct = pickle.load(open(nnDataFile, 'rb'))
    nearestNeighbours = nnDataStruct['nearestNeighbours']
    # pointIndices = nnDataStruct['pointIndices']
    adjList = nnDataStruct['adjList']
    nearNeighInitClust = nnDataStruct['nearNeighInitClust']
    initClustSubsetInd = nnDataStruct['initClustSubsetInd']
    adjListHigherDeg = nnDataStruct['adjListHigherDeg']
    print(np.max(nnDataStruct['nearestNeighbours']), pointIndices.shape[0])
    assert np.max(nnDataStruct['nearestNeighbours']) == (pointIndices.shape[0] - 1)
  else:
    raise ValueError('runPart needs to be either R or L')

  return nearestNeighbours, adjListHigherDeg, nearNeighInitClust, initClustSubsetInd


def setPrior(params, informativePrior, mean_gamma_alpha=0.7,
  std_gamma_alpha=0.01, mu_beta=0, std_beta=1):
  '''
  Sets an uninformative or informative prior, depending on the flag informativePrior
  If the prior is informtive, it uses a Gamma prior for the alpha (progression speed) and a
  Gaussian prior for beta (time shift).

  :param params:
  :param informativePrior: if true uses informative prior, otherwise uninformative
  :param mean_gamma_alpha: Mean of the Gamma prior on parameter alpha
  :param std_gamma_alpha: Std of the Gamma prior on parameter alpha
  :param mu_beta: Mean of the Gaussian prior on parameter beta
  :param std_beta: Std of the Gaussian prior on parameter beta

  std_beta = 1, assuming some subjShifts that yield std(dps_controls) == 1
  :return:
  '''
  if informativePrior:
    params['logPriorShiftFunc'] = logPriorShiftInform
    params['logPriorShiftFuncDeriv'] = logPriorShiftInformDeriv

    # make a gamma distribution for alpha with certain mean and std, then convert to shape and rate
    # mean_gamma_alpha = 1
    # std_gamma_alpha = 0.4
    shape_alpha = (mean_gamma_alpha ** 2) / (std_gamma_alpha**2)
    rate_alpha = mean_gamma_alpha / (std_gamma_alpha**2)
    print('shape_alpha', shape_alpha)
    print('rate_alpha', rate_alpha)
    print('mu_beta', mu_beta)
    print('std_beta', std_beta)
    # print(asda)
    assert (shape_alpha > 1)  # otherwise function doesn't have local maximum
    # WARNING: need to tranform the Z-score the age before I start the fit.
    params['paramsPriorShift'] = [shape_alpha, rate_alpha, mu_beta, std_beta]

    params['logPriorThetaFunc'] = logPriorThetaUnif #logPriorThetaInform
    params['logPriorThetaFuncDeriv'] = logPriorThetaUnifDeriv #logPriorThetaInformDeriv
    params['informPrior'] = True

    priorNr = 1
  else:
    # uniform prior on shifts
    params['logPriorShiftFunc'] = logPriorShiftUnif
    params['logPriorShiftFuncDeriv'] = logPriorShiftUnifDeriv
    params['paramsPriorShift'] = None

    params['logPriorThetaFunc'] = logPriorThetaUnif
    params['logPriorThetaFuncDeriv'] = logPriorThetaUnifDeriv
    params['paramsPriorTheta'] = None

    params['informPrior'] = False
    priorNr = 0

  return priorNr



def filterDDSPA(params, excludeIDlocal):
  # create data folds
  filterIndices = np.logical_not(np.in1d(params['diag'], excludeIDlocal))
  filteredParams = copy.deepcopy(params)
  filteredParams['data'] = params['data'][filterIndices,:]
  filteredParams['diag'] = params['diag'][filterIndices]
  filteredParams['scanTimepts'] = params['scanTimepts'][filterIndices]
  filteredParams['partCode'] = params['partCode'][filterIndices]
  filteredParams['ageAtScan'] = params['ageAtScan'][filterIndices]

  return filteredParams

def filterDDSPAIndices(params, filterIndices):
  # create data folds
  filteredParams = copy.deepcopy(params)
  filteredParams['data'] = params['data'][filterIndices,:]
  filteredParams['diag'] = params['diag'][filterIndices]
  filteredParams['scanTimepts'] = params['scanTimepts'][filterIndices]
  filteredParams['partCode'] = params['partCode'][filterIndices]
  filteredParams['ageAtScan'] = params['ageAtScan'][filterIndices]

  return filteredParams


def filterDDSPAIndicesShallow(params, filterIndices):
  # make a shallow copy instead, slicing should make a shallow copy in python
  filteredParams = params
  filteredParams['data'] = params['data'][filterIndices,:]
  filteredParams['diag'] = params['diag'][filterIndices]
  filteredParams['scanTimepts'] = params['scanTimepts'][filterIndices]
  filteredParams['partCode'] = params['partCode'][filterIndices]
  filteredParams['ageAtScan'] = params['ageAtScan'][filterIndices]

  return filteredParams

def calcSpatialOverlap(clustProb1, clustProb2):
  """
  calculates spatial overlap between two cluster probability arrays. Finds the optimal permutation,
  then performs argmax for every vertex to find the most likely cluster it belongs to. In future work,
  this will be extended to account for all the probabilities.

  :param clustProb1:
  :param clustProb2:
  :return:
  """
  nrBiomk, nrClust = clustProb1.shape
  clustAss2 = np.argmax(clustProb2, axis=1)

  currPerm = np.array(range(nrClust))
  clustProbPermed1 = clustProb1[:, currPerm]
  clustAssPermed1 = np.argmax(clustProbPermed1, axis=1)

  currCorrectAssignMean = np.mean(clustAssPermed1 == clustAss2)

  nrIt = 100

  for i in range(nrIt):
    # print('permutation', perm)
    newPerm = perturbPerm(currPerm)
    clustProbPermed1 = clustProb1[:, newPerm]
    clustAssPermed1 = np.argmax(clustProbPermed1, axis=1)
    newCorrectAssign = np.mean(clustAssPermed1 == clustAss2)

    if newCorrectAssign > currCorrectAssignMean:
      currPerm = newPerm
      currCorrectAssignMean = newCorrectAssign

      print('found better perm', newCorrectAssign, newPerm)

  clustProbPermed1 = clustProb1[:, currPerm]
  clustAssPermed1 = np.argmax(clustProbPermed1, axis=1)

  dice = calcDiceOverlap(clustAssPermed1, clustAss2, nrClust)

  print('currCorrectAssignMean, np.array(currPerm), dice', currCorrectAssignMean, np.array(currPerm), dice)

  return currCorrectAssignMean, np.array(currPerm), dice

def perturbPerm(currPerm):
  from random import randint
  nrClust = currPerm.shape[0]
  srcPos = randint(0, nrClust - 1)
  trgPos = randint(0, nrClust - 1)

  newPerm = copy.deepcopy(currPerm)

  auxVal = newPerm[trgPos]
  newPerm[trgPos] = newPerm[srcPos]
  newPerm[srcPos] = auxVal

  # print('newPerm', newPerm, '  currPerm', currPerm)

  return newPerm

def calcDiceOverlap(maxLikClustB1, maxLikClustB2, nrClust):
  # maxLikClustB1 = np.argmax(clustProb1, axis=1)
  # maxLikClustB2 = np.argmax(clustProb2, axis=1)
  # nrBiomk, nrClust = clustProb1.shape

  dice = np.zeros(nrClust, float)
  for c in range(nrClust):
    # find union - nr of vertices that match
    union = np.sum(np.logical_and(maxLikClustB1 == c, maxLikClustB2 == c))
    # divide by nr_vertices in image

    dice[c] = 2 * union / (np.sum(maxLikClustB1 == c) +
                           np.sum(maxLikClustB2 == c))


  for c in range(nrClust):
    print('avg dice for c %d' % c, dice[c])

  avgDiceOverall = np.mean(dice)

  print('avgDiceOverall', avgDiceOverall)

  return avgDiceOverall