import numpy as np
from matplotlib import pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as a3
#pl.ioff()
from itertools import cycle
from matplotlib.lines import Line2D
import matplotlib.colors
import matplotlib
import nibabel as nib
import colorsys
import scipy
import os

from aux import *


def plotClust3DBrain(clustProbBC, plotTrajParams):
  """
  3D brain mesh from the set of triangles from lh.pial

  :param clustProbBC:
  :param plotTrajParams:
  :return:
  """

  pointIndices = plotTrajParams['pointIndices']
  fsaveragePialLh = '%s/subjects/fsaverage/surf/lh.pial' % plotTrajParams['freesurfPath']
  coordsLh, facesLh, _ = nib.freesurfer.io.read_geometry(fsaveragePialLh, read_metadata = True)

  fsaveragePialRh = '%s/subjects/fsaverage/surf/rh.pial' % plotTrajParams['freesurfPath']
  coordsRh, facesRh, _ = nib.freesurfer.io.read_geometry(fsaveragePialRh, read_metadata = True)

  #print(coordsLh.shape, coordsLh[1:30,:])

  print(np.sum(clustProbBC,1))
  assert(all(np.abs(np.sum(clustProbBC,1) - 1) < 0.001))

  nrBiomk, nrClust = clustProbBC.shape
  clustHuePoints = plotTrajParams['clustHuePoints']
  print(clustHuePoints)

  colsB = []
  colsC = []

  for b in range(nrBiomk): # nr points
    hue = clustHuePoints[np.argmax(clustProbBC[b,:])]
    colsB += [colorsys.hsv_to_rgb(hue, 1, 1)]

  for c in range(nrClust):
    hue = clustHuePoints[c]
    colsC += [colorsys.hsv_to_rgb(hue, 1, 1)]

  pl.clf()
  fig = pl.figure(2)
  ax = fig.add_subplot(111, projection='3d')

  markerSize = 20

  nrTriangles = facesLh.shape[0]

  # print(np.max(coordsLh[:,0]), np.min(coordsLh[:,0]))
  # print(np.max(coordsLh[:,1]), np.min(coordsLh[:,1]))
  # print(np.max(coordsLh[:,2]), np.min(coordsLh[:,2]))

  coordScaleFactor = np.max(np.abs([np.max(coordsLh), np.min(coordsLh)]))
  print(coordScaleFactor)

  perms3 = [[0,1,2], [0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]

  coordsLh /= coordScaleFactor

  triangles = [coordsLh[facesLh[p,perms3[0]],:] for p in range(nrTriangles)]
  #triangles = [scipy.rand(3,3) for p in range(nrTriangles)]

  #print([triangles[i] for i in range(5)])
  ax.add_collection3d(a3.art3d.Poly3DCollection(triangles, facecolors='b', linewidths=0))

  # for p in range(nrTriangles):
  #   tri = a3.art3d.Poly3DCollection([coordsLh[facesLh[p,:],:]])
  #   #tri.set_color(matplotlib.colors.rgb2hex())
  #   tri.set_edgecolor('k')
  #   ax.add_collection3d(tri)


  # h = ax.scatter(xs=coordsLh[pointIndices,0], ys=coordsLh[pointIndices,1],
  #   zs=coordsLh[pointIndices,2], zdir='z', s=markerSize, c=colsB)

  adjustCurrFig(plotTrajParams)

  mng = pl.get_current_fig_manager()
  mng.resize(*plotTrajParams['Clust3DMaxWinSize'])

  legCircles = [pl.Line2D([0, 0], [0, 1], color=col, marker='o', linestyle='') for col in colsC]
  pl.legend(legCircles, ['clust %d' % c for c in range(nrClust)], loc=4)

  # pl.show()
  # print(sdsa)
  fig.show()

  return fig

def plotScoresHist(scores, labels):

  nrGroups = len(scores)
  assert len(labels) == nrGroups

  means = [np.mean(g) for g in scores]
  stds = [np.std(g) for g in scores]

  fig = pl.figure(1)
  width = 0.35
  groupCols = ['r', 'g', 'b']
  for g in range(nrGroups):
    rects1 = pl.bar(g*width, means[g], width, color=groupCols[g], yerr=stds[g])

  ax = pl.gca()
  ax.set_xticklabels(labels)  # remove xticklabels


  fig.show()


def plotClust3DScatter(clustProbBC, plotTrajParams):
  '''
  3D scatter plot of points on the brain surface

  :param clustProbBC:
  :param plotTrajParams:
  :return:
  '''
  pointIndices = plotTrajParams['pointIndices']
  fsaveragePialLh = '%s/subjects/fsaverage/surf/lh.pial' % plotTrajParams['freesurfPath']
  coordsLh, _, _ = nib.freesurfer.io.read_geometry(fsaveragePialLh, read_metadata = True)

  fsaveragePialRh = '%s/subjects/fsaverage/surf/rh.pial' % plotTrajParams['freesurfPath']
  coordsRh, _, _ = nib.freesurfer.io.read_geometry(fsaveragePialRh, read_metadata = True)

  #print(coordsLh.shape, coordsLh[1:30,:])

  print(np.sum(clustProbBC,1))
  assert(all(np.abs(np.sum(clustProbBC,1) - 1) < 0.001))

  nrBiomk, nrClust = clustProbBC.shape
  clustHuePoints = plotTrajParams['clustHuePoints']
  print(clustHuePoints)

  colsB = []
  colsC = []

  for b in range(nrBiomk): # nr points
    hue = clustHuePoints[np.argmax(clustProbBC[b,:])]
    colsB += [colorsys.hsv_to_rgb(hue, 1, 1)]

  for c in range(nrClust):
    hue = clustHuePoints[c]
    colsC += [colorsys.hsv_to_rgb(hue, 1, 1)]

  pl.clf()
  fig = pl.figure(2)
  ax = fig.add_subplot(111, projection='3d')

  markerSize = 20

  h = ax.scatter(xs=coordsLh[pointIndices,0], ys=coordsLh[pointIndices,1],
    zs=coordsLh[pointIndices,2], zdir='z', s=markerSize, c=colsB)
  # ax.scatter(xs=coordsRh[pointIndices, 0], ys=coordsRh[pointIndices, 1],
  #   zs=coordsRh[pointIndices, 2], zdir='z', s=markerSize, c=cols)

  adjustCurrFig(plotTrajParams)

  mng = pl.get_current_fig_manager()
  mng.resize(*plotTrajParams['Clust3DMaxWinSize'])

  legCircles = [pl.Line2D([0, 0], [0, 1], color=col, marker='o', linestyle='') for col in colsC]
  pl.legend(legCircles, ['clust %d' % c for c in range(nrClust)], loc=4)

  # pl.show()
  # print(sdsa)
  fig.show()

  return fig


def visData(data, diag, age, plotTrajParams, sortedByPvalInd):
  '''
  Plots average biomarker value for various ROIs

  :param data: NR_CROSS_SUBJ x NR_BIOMK array
  :param diag: NR_CROSS_SUBJ x 1
  :param age:  NR_CROSS_SUBJ x 1
  :param plotTrajParams: dictionary of plotting parameters
  :param sortedByPvalInd: ROI indicesof each point on the surface, sorted by p-value (the regions for which we observe the highest differences between CTL and AD apprear first)

  :return: figure handle
  '''

  pointIndices = plotTrajParams['pointIndices']
  labels = plotTrajParams['labels']
  names = plotTrajParams['names']

  fig = pl.figure()
  nrRows = 3
  nrCols = 4
  nrBiomkToDisplay = nrRows * nrCols

  nrSubj, nrBiomk = data.shape

  xs = np.linspace(np.min(age), np.max(age), 100)
  #diagNrs = np.unique(diag)
  diagNrs = plotTrajParams['diagNrs']
  #print(diagNrs)
  #print(asdsa)

  nrSubjToDisplay = nrSubj

  dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)

  dataSubset = data[dataSubsetIndices, :]
  diagSubset = diag[dataSubsetIndices]
  ageSubset = age[dataSubsetIndices]

  if sortedByPvalInd is None:
    biomkIndices = range(nrBiomk)[::10]
  else:
    biomkIndices = sortedByPvalInd

  print('biomkIndices', biomkIndices)

  counter = 0
  for b in biomkIndices:

    print(b)

    row = np.divide(b,nrRows)
    col = np.mod(b,nrCols)

    ax = pl.subplot(nrRows, nrCols, 1+np.mod(counter, nrBiomkToDisplay))
    #print(pointIndices[b])
    #print(labels)
    print(labels[pointIndices[b]])

    ax.set_title('vertex %s' % names[labels[pointIndices[b]]] )

    #lines = []
    for d in range(len(diagNrs)):
      pl.scatter(ageSubset[diagSubset == diagNrs[d]],
        dataSubset[diagSubset == diagNrs[d],b], s=20,
                 c=plotTrajParams['diagColors'][diagNrs[d]],
        label=plotTrajParams['diagLabels'][diagNrs[d]])

    if col == 0:
      pl.ylabel('$Z-score')
    if row == nrRows - 1:
      pl.xlabel('$dps$')

    #tMin, tMax = plotTrajParams['xLim']
    # print tMin, tMax
    pl.xlim(np.min(age), np.max(age))
    pl.ylim(np.min(dataSubset[:,b]), np.max(dataSubset[:,b]))

    if counter == 0:
      adjustCurrFig(plotTrajParams)
      fig.suptitle('indiv points', fontsize=20)

      h, axisLabels = ax.get_legend_handles_labels()
      #print(h[2:4], labels[2:4])
      #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
      #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

      legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
      #print(legend.legendHandles)
      #print(asdsa)
      # set the linewidth of each legend object
      for i,legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(4.0)
        legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

      mng = pl.get_current_fig_manager()
      mng.resize(*plotTrajParams['SubfigVisMaxWinSize'])

    #print(np.mod(counter,nrBiomkToDisplay))
    if np.mod(counter,nrBiomkToDisplay) == 0:
      fig.show()

      print("Plotting results .... ")
      pl.pause(10)

    counter += 1

  return fig


def visRegions(data, diag, age, plotTrajParams):
  pointIndices = plotTrajParams['pointIndices']
  labels = plotTrajParams['labels']
  names = plotTrajParams['names']

  fig = pl.figure()
  nrRows = 3
  nrCols = 4
  nrBiomkToDisplay = nrRows * nrCols

  nrSubj, nrBiomk = data.shape
  nrRegions = len(names)

  xs = np.linspace(np.min(age), np.max(age), 100)
  #diagNrs = np.unique(diag)

  diagNrs = plotTrajParams['diagNrs']


  #print(diagNrs)
  #print(asdsa)

  counter = 0
  dataMeanByRegion = np.zeros((data.shape[0], nrRegions), float)

  for r in range(nrRegions):

    print(r)

    row = np.divide(r,nrRows)
    col = np.mod(r,nrCols)

    ax = pl.subplot(nrRows, nrCols, 1+np.mod(counter, nrBiomkToDisplay))
    dataMeanByRegion[:,r] = np.mean(data[:,labels[pointIndices] == r],axis=1)

    ax.set_title('region %s' % names[r] )

    for d in range(len(diagNrs)):
      pl.scatter(age[diag == diagNrs[d]], dataMeanByRegion[diag == diagNrs[d],r], s=20,
        c=plotTrajParams['diagColors'][diagNrs[d]],
        label=plotTrajParams['diagLabels'][diagNrs[d]])

    if col == 0:
      pl.ylabel('$Z-score$')
    if row == nrRows - 1:
      pl.xlabel('$dps$')

    #tMin, tMax = plotTrajParams['xLim']
    # print tMin, tMax
    pl.xlim(np.min(age), np.max(age))
    pl.ylim(np.min(dataMeanByRegion[:,r]), np.max(dataMeanByRegion[:,r]))

    if counter == 0:
      adjustCurrFig(plotTrajParams)
      fig.suptitle('indiv points', fontsize=20)

      h, axisLabels = ax.get_legend_handles_labels()
      #print(h[2:4], labels[2:4])
      #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
      #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

      legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
      #print(legend.legendHandles)
      #print(asdsa)
      # set the linewidth of each legend object
      for i,legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(4.0)
        legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

      mng = pl.get_current_fig_manager()
      mng.resize(*plotTrajParams['SubfigVisMaxWinSize'])

    #print(np.mod(counter,nrBiomkToDisplay))
    if np.mod(counter,nrBiomkToDisplay) == (nrBiomkToDisplay-1):
      fig.show()

      print("Plotting results .... ")
      pl.pause(30)

    counter += 1

  return fig



def adjustCurrFig(plotTrajParams):
  fig = matplotlib.pyplot.gcf()
  #fig.set_size_inches(180/fig.dpi, 100/fig.dpi)

  mng = pl.get_current_fig_manager()
  if plotTrajParams['agg']: # if only printing images
    pass
  else:
    maxSize = mng.window.maxsize()
    maxSize = (maxSize[0]/2.1, maxSize[1]/1.1)
    #print(maxSize)
    mng.resize(*maxSize)

    #mng.window.SetPosition((500, 0))
    mng.window.wm_geometry("+200+50")

  #pl.tight_layout()
  pl.gcf().subplots_adjust(bottom = 0.25)

  #pl.tight_layout(pad=50, w_pad=25, h_pad=25)


def plotStagingConsist(plotTrajParams, longAgeAtScan, dpsLong, longDiag):
  (meanAgeCTL, stdAgeCTL) = plotTrajParams['ageTransform']

  fig = pl.figure()

  nrSubjLong = len(longAgeAtScan)
  assert(len(longDiag) == nrSubjLong)

  unqDiags = np.unique(longDiag)
  nrUnqDiag = np.max(unqDiags)

  diagCounters = [0 for x in range(nrUnqDiag)]
  legendHandles = []

  for s in range(nrSubjLong):
    legLabel = None
    if diagCounters[longDiag[s] - 1] == 0:
      # legendHandles += [line]
      legLabel = plotTrajParams['diagLabels'][longDiag[s]]

    line, = pl.plot(longAgeAtScan[s]*stdAgeCTL+meanAgeCTL, dpsLong[s],
        '%s-' % (plotTrajParams['diagColors'][longDiag[s]]),label=legLabel)

    diagCounters[longDiag[s] - 1] += 1

  # h, axisLabels = ax.get_legend_handles_labels()

  print(legendHandles)
  pl.legend(loc='upper left')
  # legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )

  pl.xlabel('age')
  pl.ylabel('DPS')

  mng = pl.get_current_fig_manager()
  size = (800,400)
  mng.resize(*size)


  fig.show()

  # also calculate number of subjects with increasing DPS scores
  assert nrSubjLong == len(dpsLong)
  consistStagingBinArray = np.zeros(nrSubjLong, bool)
  for s in range(nrSubjLong):
    consistStagingBinArray[s] = dpsLong[s][1] >= dpsLong[s][0]

  print('unqDiags', unqDiags)

  for diag in range(nrUnqDiag)[:3]:
    nrConsistCurrDiag = np.sum(consistStagingBinArray[longDiag == unqDiags[diag]])
    totalNrCurrDiag = np.sum(longDiag == unqDiags[diag])
    print('%s: %.2f   %d  %d' % (plotTrajParams['diagLabels'][unqDiags[diag]],
    nrConsistCurrDiag/totalNrCurrDiag, nrConsistCurrDiag, totalNrCurrDiag))

  nrSubjConsistStaging = float(np.sum([1 for s in dpsLong if s[1] >= s[0]]))
  print('nr of subj with consistent staging:', nrSubjConsistStaging)
  print('proportion with consistent staging', nrSubjConsistStaging / len(dpsLong))

  return fig


def plotStagingHist(maxLikStages, diag, plotTrajParams, expNameCurrModel):
  '''
  Plots staging histogram for all diagnostic groups in 'diag'

  Parameters
  ----------
  maxLikStages
  nrStages
  diag
  plotTrajParams
  expNameCurrModel - unique identifier for the current experiment

  Returns
  -------
  fig - figure handle
  legend - legend handle

  '''

  fig = pl.figure()
  diagNrs = np.unique(diag)
  print(plotTrajParams['diagLabels'])
  colors = [plotTrajParams['diagColors'][d] for d in diagNrs]
  legendEntries = [plotTrajParams['diagLabels'][d] for d in diagNrs]
  histObj = pl.hist([maxLikStages[diag == d] for d in diagNrs],
    bins=plotTrajParams['stagingHistNrBins'], color=colors, label=legendEntries)
  lgd = pl.legend(loc='upper right', ncol=plotTrajParams['legendCols'])
  fig.suptitle('%s staging' % expNameCurrModel, fontsize=20)

  axes = pl.gca()
  yLimCurr = axes.get_ylim()

  pl.ylim(yLimCurr[0], yLimCurr[1]+(yLimCurr[1]-yLimCurr[0])/5)

  fig.show()
  #print(adsa)
  return fig, lgd
