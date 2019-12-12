import numpy as np
from matplotlib import pyplot as pl
import matplotlib
import colorsys
from plotFunc import *
import scipy
import aux
import numpy.ma as ma

class PlotterVDPM:

  ''' standard class for generating a brain colouring of the cluters estimated by the model'''

  def __init__(self):
    pass

  def plotTrajSubfigWithDataRandPoints(self, data, diag, dps, thetas, variances, prevClustProbBCColNorm,
                                       plotTrajParams, trajFunc, replaceFigMode=True,
                                       thetasSamplesClust=None, colorTitle=True):


    figSizeInch = (plotTrajParams['SubfigClustMaxWinSize'][0]/100,
    plotTrajParams['SubfigClustMaxWinSize'][1] / 100)
    fig = pl.figure(3, figsize=figSizeInch)
    pl.clf()
    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']

    nrSubj, nrBiomk = data.shape
    nrClust = prevClustProbBCColNorm.shape[1]

    print('nrClust', nrClust)
    print('prevClustProbBCColNorm.shape',prevClustProbBCColNorm.shape)
    print('nrRows', nrRows)
    print('nrCols', nrCols)
    print('plotTrajParams[clustCols]', plotTrajParams['clustCols'])


    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj
    nrBiomkToDisplay = nrSubj # take one random point for each subject
    np.random.seed(1)

    dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)
    clustSubsetIndices = [np.random.choice(np.array(range(nrBiomk)), nrBiomkToDisplay, replace = True, p=prevClustProbBCColNorm[:,c]) for c in range(nrClust)]

    dataSubset = data[dataSubsetIndices, :]
    diagSubset = diag[dataSubsetIndices]
    dpsSubset = dps[dataSubsetIndices]

    for row in range(nrRows):
      for col in range(nrCols):
        c = row * nrCols + col # clusterNr
        if c < nrClust:
          ax = pl.subplot(nrRows, nrCols, c + 1)
          ax.set_title('cluster %s' % c)
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][c])

          fsCurr = trajFunc(xs, thetas[c,:])

          pl.plot(xs, fsCurr, 'k-', linewidth=3.0) # label='sigmoid traj %d' % c
          pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[c],
            (fsCurr + 1.9600 * stdDevs[c])[::-1]]), alpha=.3, fc='b', ec='None')
                  #label='conf interval (1.96*std)')

          #lines = []
          for d in range(len(diagNrs)):
            #print(clustSubsetIndices, dataSubset[diagSubset == diagNrs[d], clustSubsetIndices[c]].shape)
            pointsSubsetCurrDiag = clustSubsetIndices[c][diagSubset == diagNrs[d]]
            pl.scatter(dpsSubset[diagSubset == diagNrs[d]],
              dataSubset[diagSubset == diagNrs[d],pointsSubsetCurrDiag],
              s=20, c=plotTrajParams['diagColors'][diagNrs[d]],
              label=plotTrajParams['diagLabels'][diagNrs[d]])

          if thetasSamplesClust is not None:
            for t in range(thetasSamplesClust[c].shape[0]):
              pl.plot(xs, trajFunc(xs, thetasSamplesClust[c][t, :]))

          if col == 0:
            pl.ylabel('$Z-score\ of\ biomarker$')
          if row == nrRows - 1:
            pl.xlabel('$disease\ progression\ score$')

          pl.xlim(np.min(dps), np.max(dps))
          if 'ylimitsRandPoints' in plotTrajParams.keys():
            # print(asds)
            pl.ylim(plotTrajParams['ylimitsRandPoints'][0], plotTrajParams['ylimitsRandPoints'][1])
          else:
            pl.ylim(np.min(dataSubset),
                    np.max(dataSubset))


    # adjustCurrFig(plotTrajParams)
    fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    #print(h[2:4], labels[2:4])
    #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # set the linewidth of each legend object
    for i,legobj in enumerate(legend.legendHandles):
      legobj.set_linewidth(4.0)
      legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

    # mng = pl.get_current_fig_manager()
    # mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    # print(ads)

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    print("Plotting results .... ")
    pl.pause(5.05)

    return fig

  def plotTrajSubfigGivenVoxel(self, data, diag, dps, thetas, variances, clustProbC, plotTrajParams,
                               trajFunc, voxNr, colorTitle=True):


    fig = pl.figure(1)
    pl.clf()
    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']

    nrSubj, nrBiomk = data.shape
    nrClust = clustProbC.shape[0]
    bestClust = np.argmax(clustProbC)

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj
    nrBiomkToDisplay = 1

    dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)

    dataSubset = data[dataSubsetIndices, voxNr]
    diagSubset = diag[dataSubsetIndices]
    dpsSubset = dps[dataSubsetIndices]

    for row in range(nrRows):
      for col in range(nrCols):
        c = row * nrCols + col # clusterNr
        if c < nrClust:
          ax = pl.subplot(nrRows, nrCols, c + 1)
          if bestClust == c:
            ax.set_title('cluster %s - best' % c)
          else:
            ax.set_title('cluster %s' % c)
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][c])

          fsCurr = trajFunc(xs, thetas[c,:])

          pl.plot(xs, fsCurr, 'k-', linewidth=3.0) # label='sigmoid traj %d' % c
          pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[c],
            (fsCurr + 1.9600 * stdDevs[c])[::-1]]), alpha=.3, fc='b', ec='None')
                  #label='conf interval (1.96*std)')


          #lines = []
          for d in range(len(diagNrs)):
            #print(clustSubsetIndices, dataSubset[diagSubset == diagNrs[d], clustSubsetIndices[c]].shape)
            pl.scatter(dpsSubset[diagSubset == diagNrs[d]],
              dataSubset[diagSubset == diagNrs[d]], s=20,
              c=plotTrajParams['diagColors'][diagNrs[d]],
              label=plotTrajParams['diagLabels'][diagNrs[d]])

          if col == 0:
            pl.ylabel('$Z-score\ of\ biomarker$')
          if row == nrRows - 1:
            pl.xlabel('$disease\ progression\ score$')

          pl.xlim(np.min(dps), np.max(dps))
          pl.ylim(np.min(dataSubset), np.max(dataSubset))

    adjustCurrFig(plotTrajParams)
    fig.suptitle('voxel %d' % voxNr, fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    #print(h[2:4], labels[2:4])
    #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # set the linewidth of each legend object
    for i,legobj in enumerate(legend.legendHandles):
      legobj.set_linewidth(4.0)
      legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

    mng = pl.get_current_fig_manager()
    mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    pl.show()

    print("Plotting results .... ")
    pl.pause(0.05)

    return fig

  def plotTrajSubfigWithData(self, data, diag, dps, thetas, variances, prevClustProbBCColNorm,
                             plotTrajParams, trajFunc, colorTitle=True):


    fig = pl.figure(1)
    pl.clf()
    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']

    nrSubj, nrBiomk = data.shape
    nrClust = prevClustProbBCColNorm.shape[1]

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj
    nrBiomkToDisplay = 1
    np.random.seed(1)

    dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)
    clustSubsetIndices = [np.random.choice(np.array(range(nrBiomk)), nrBiomkToDisplay, replace = False, p=prevClustProbBCColNorm[:,c]) for c in range(nrClust)]

    dataSubset = data[dataSubsetIndices, :]
    diagSubset = diag[dataSubsetIndices]
    dpsSubset = dps[dataSubsetIndices]

    for row in range(nrRows):
      for col in range(nrCols):
        c = row * nrCols + col # clusterNr
        if c < nrClust:
          ax = pl.subplot(nrRows, nrCols, c + 1)
          ax.set_title('cluster %s' % c)
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][c])

          fsCurr = trajFunc(xs, thetas[c,:])

          pl.plot(xs, fsCurr, 'k-', linewidth=3.0) # label='sigmoid traj %d' % c
          pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[c],
            (fsCurr + 1.9600 * stdDevs[c])[::-1]]), alpha=.3, fc='b', ec='None')
                  #label='conf interval (1.96*std)')


          #lines = []
          for d in range(len(diagNrs)):
            #print(clustSubsetIndices, dataSubset[diagSubset == diagNrs[d], clustSubsetIndices[c]].shape)
            pl.scatter(dpsSubset[diagSubset == diagNrs[d]],
              dataSubset[diagSubset == diagNrs[d],:][:,clustSubsetIndices[c]],
              s=20, c=plotTrajParams['diagColors'][diagNrs[d]],
              label=plotTrajParams['diagLabels'][diagNrs[d]])

          if col == 0:
            pl.ylabel('$Z-score\ of\ biomarker$')
          if row == nrRows - 1:
            pl.xlabel('$disease\ progression\ score$')

          pl.xlim(np.min(dps), np.max(dps))
          # pl.ylim(np.min(dataSubset[:,clustSubsetIndices[c]]),
          #   np.max(dataSubset[:, clustSubsetIndices[c]]))
          pl.ylim(-5,5)

    adjustCurrFig(plotTrajParams)
    fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    #print(h[2:4], labels[2:4])
    #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # set the linewidth of each legend object
    for i,legobj in enumerate(legend.legendHandles):
      legobj.set_linewidth(4.0)
      legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

    mng = pl.get_current_fig_manager()
    mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    fig.show()

    print("Plotting results .... ")
    pl.pause(0.05)

    return fig


  def plotTrajWeightedDataMean(self, data, diag, dps, longData, longDiag, longDPS, thetas, variances, clustProbBCColNorm, plotTrajParams,
                               trajFunc, replaceFigMode=True, thetasSamplesClust=None, showConfInt=True,
                               colorTitle=True, yLimUseData=False, adjustBottomHeight=0.25, orderClust=False):

    figSizeInch = (plotTrajParams['SubfigClustMaxWinSize'][0]/100,
    plotTrajParams['SubfigClustMaxWinSize'][1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']


    nrSubj, nrBiomk = data.shape
    nrClust = clustProbBCColNorm.shape[1]

    if orderClust:
      clustOrderInd = aux.orderTrajBySlope(thetas)
    else:
      clustOrderInd = range(nrClust)

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj
    dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)

    dataSubset = data[dataSubsetIndices, :]
    diagSubset = diag[dataSubsetIndices]
    dpsSubset = dps[dataSubsetIndices]

    nrSubjLong = len(longData)
    weightedDataMean = np.zeros((nrSubj, nrClust))
    weightedDataMeanLong = [np.zeros((longData[s].shape[0], nrClust),float) for s in range(nrSubjLong)]

    for c in range(nrClust):
      # compute weighted mean of data
      weightedDataMean[:, c] = np.average(dataSubset, axis=1, weights=clustProbBCColNorm[:, c])

      for s in range(nrSubjLong):
        weightedDataMeanLong[s][:,c] = np.average(longData[s], axis=1, weights=clustProbBCColNorm[:, c])

    minY = np.min(weightedDataMean, axis=(0,1))
    maxY = np.max(weightedDataMean, axis=(0,1))
    delta = (maxY - minY) / 10

    for row in range(nrRows):
      for col in range(nrCols):
        c = row * nrCols + col # clusterNr
        if c < nrClust:
          ax = pl.subplot(nrRows, nrCols, clustOrderInd[c] + 1)
          ax.set_title('cluster %s' % clustOrderInd[c])
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][c])

          fsCurr = trajFunc(xs, thetas[c,:])

          #print(fsCurr.shape, xs.shape)
          pl.plot(xs, fsCurr, 'k-', linewidth=3.0) # label='sigmoid traj %d' % c
          if showConfInt:
            pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[c],
              (fsCurr + 1.9600 * stdDevs[c])[::-1]]), alpha=.3, fc='b', ec='None')
                    #label='conf interval (1.96*std)')

          if thetasSamplesClust is not None:
            for t in range(thetasSamplesClust[c].shape[0]):
              pl.plot(xs, trajFunc(xs, thetasSamplesClust[c][t, :]))


          ############# scatter plot subjects ######################
          # for d in range(len(diagNrs)):
          #   #print(clustSubsetIndices, dataSubset[diagSubset == diagNrs[d], clustSubsetIndices[c]].shape)
          #   pl.scatter(dpsSubset[diagSubset == diagNrs[d]], weightedDataMean[diagSubset == diagNrs[d],c],
          #   s=20, c=plotTrajParams['diagColors'][diagNrs[d]],
          #     label=plotTrajParams['diagLabels'][diagNrs[d]])

          ############# spagetti plot subjects ######################
          counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
          for s in range(nrSubjLong):
            #print(clustSubsetIndices, dataSubset[diagSubset == diagNrs[d], clustSubsetIndices[c]].shape)
            labelCurr = None
            if counterDiagLegend[longDiag[s]] == 0:
              labelCurr = plotTrajParams['diagLabels'][longDiag[s]]
              counterDiagLegend[longDiag[s]] += 1

            pl.plot(longDPS[s], weightedDataMeanLong[s][:,c],
              c=plotTrajParams['diagColors'][longDiag[s]],
              label=labelCurr)

          if col == 0:
            pl.ylabel('$Z-score\ of\ biomarker$')
          if row == nrRows - 1:
            pl.xlabel('$disease\ progression\ score$')

          pl.xlim(np.min(dps), np.max(dps))

          # pl.ylim(-1.6,1)
          if 'ylimTrajWeightedDataMean' in plotTrajParams.keys() and not yLimUseData:
            pl.ylim(plotTrajParams['ylimTrajWeightedDataMean'])
          else:
            pl.ylim(minY - delta, maxY + delta)

    adjustCurrFig(plotTrajParams)
    pl.gcf().subplots_adjust(bottom=adjustBottomHeight)
    #fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    #print(h[2:4], labels[2:4])
    #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # set the linewidth of each legend object
    # for i,legobj in enumerate(legend.legendHandles):
    #   legobj.set_linewidth(4.0)
    #   legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

    mng = pl.get_current_fig_manager()
    # print(plotTrajParams['SubfigClustMaxWinSize'])
    # print(adsds)
    mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    #print("Plotting results .... ")
    pl.pause(0.05)
    return fig

  def plotTrajSamplesAndHist(self, data, diag, dps, thetas, variances, clustProbBCColNorm, plotTrajParams,
     trajFunc, replaceFigMode=True, thetasSamplesClust=None, showConfInt=True,
     colorTitle=True, yLimUseData=False, adjustBottomHeight=0.25, orderClust=False,
     windowSize=(600,800)):

    fontSize = 12
    figSizeInch = (windowSize[0] / 100, windowSize[1] / 100)

    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    nrVirtualSubfigs = 4

    dpsSorted = np.sort(dps)
    quantile = 0.98
    minX = dpsSorted[int(dpsSorted.shape[0]*(1-quantile))]
    maxX = dpsSorted[int(dpsSorted.shape[0]*quantile)]
    deltaX = (maxX - minX) / 10


    ################ make top plot ##################
    ax0 = pl.subplot2grid((nrVirtualSubfigs, 1), (0, 0), rowspan=1)
    # fig.subplots_adjust(bottom=0.4)
    # ax0.set_frame_on(False)
    # fig.subplots_adjust(top=0.75)
    diagNrs = np.unique(diag)
    print('diagNrs, diag', diagNrs, diag)

    for d in range(diagNrs.shape[0]):
      labelCurr = plotTrajParams['diagLabels'][diagNrs[d]]
      colCurr = plotTrajParams['diagColors'][diagNrs[d]]

      kernel = scipy.stats.gaussian_kde(dps[diag == diagNrs[d]])
      #ysKernel = yLim[0] + kernel(xs) * (yLim[1] - yLim[0])
      ysKernel = kernel(xs)
      pl.plot(xs, ysKernel, '-', linewidth=2.0, color=colCurr, label=labelCurr)


    ax0.set_xticklabels([''] * 100) # remove xticklabels
    if plotTrajParams['datasetFull'] == 'adniPet':
      legend = pl.legend(bbox_to_anchor=(-0.1, 1.02, 1.2, .102), loc=3, mode='expand', ncol=5)
    else:
      legend = pl.legend(bbox_to_anchor=(-0.1, 1.02, 1.2, .52), loc='center', ncol=5)

    # legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # # set the linewidth of each legend object
    for i,legobj in enumerate(legend.legendHandles):
      legobj.set_linewidth(4.0)

    pl.xlim(minX-deltaX, maxX+deltaX)

    ################ make bottom plot ##################
    ax1 = pl.subplot2grid((nrVirtualSubfigs, 1), (1, 0), rowspan=(nrVirtualSubfigs - 1))

    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']

    nrSubj, nrBiomk = data.shape
    nrClust = clustProbBCColNorm.shape[1]

    if orderClust:
      clustOrderInd = aux.orderTrajBySlope(thetas)
    else:
      clustOrderInd = range(nrClust)

    # nrSubjToDisplay = nrSubj
    # dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)

    # data = data[dataSubsetIndices, :]
    # diag = diag[dataSubsetIndices]
    # dps = dps[dataSubsetIndices]

    weightedDataMean = np.zeros((nrSubj, nrClust))

    for c in range(nrClust):
      # compute weighted mean of data
      weightedDataMean[:, c] = np.average(data, axis=1, weights=clustProbBCColNorm[:, c])

    # minY = np.min(weightedDataMean, axis=(0,1))
    # maxY = np.max(weightedDataMean, axis=(0,1))
    sortedData = np.sort(weightedDataMean.reshape(-1))
    quantile = 0.99
    minY = sortedData[int(sortedData.shape[0]*(1-quantile))]
    maxY = sortedData[int(sortedData.shape[0]*quantile)]
    delta = (maxY - minY) / 10

    # color cluster traj according to slopes
    clustColsRGB, clustColsRGBperturb, slopesSortedInd = colorClustBySlope(thetas)

    if nrClust > 8:
      legIndToPlot = list(np.array(range(nrClust))[::int(nrClust/5)])
      if legIndToPlot[-1] != (nrClust-1):
        # add last cluster if it doesn't exist
        legIndToPlot = list(legIndToPlot) + [nrClust-1]
    else:
      legIndToPlot = range(nrClust)

    for c in range(nrClust):

      #print(xs[50:60], fsCurr[50:60], thetas[c,:])
      #print(asda)

      clustInd = slopesSortedInd[c]
      if thetasSamplesClust is not None:
        for t in range(thetasSamplesClust[clustInd].shape[0]):
          pl.plot(xs, trajFunc(xs, thetasSamplesClust[clustInd][t, :]), color=clustColsRGBperturb[c])

      fsCurr = trajFunc(xs, thetas[clustInd, :])
      # print(fsCurr.shape, xs.shape)
      if c in legIndToPlot:
        label = 'cluster %d' % c
      else:
        label = None
      pl.plot(xs, fsCurr, '-', linewidth=3.0, color=clustColsRGB[c], label=label)  # label='sigmoid traj %d' % c

    pl.ylabel(plotTrajParams['biomkAxisLabel'], fontsize=fontSize)
    pl.xlabel('DPS', fontsize=fontSize)
    pl.tick_params(axis='both', which='major', labelsize=fontSize)
    pl.tick_params(axis='both', which='minor', labelsize=fontSize)

    pl.xlim(minX-deltaX, maxX+deltaX)
    # pl.ylim(-1.6,1)

    if yLimUseData:
      yLim = [minY - delta, maxY + delta]
    else:
      yLim = plotTrajParams['ylimTrajSamplesInOneNoData']

    pl.ylim(yLim[0], yLim[1])

    # adjustCurrFig(plotTrajParams)
    pl.gcf().subplots_adjust(left=adjustBottomHeight, bottom=adjustBottomHeight)
    #fig.suptitle('cluster trajectories', fontsize=20)

    # h, axisLabels = ax.get_legend_handles_labels()
    # #print(h[2:4], labels[2:4])
    # #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    # #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.legend(loc='upper right')

    # legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # # set the linewidth of each legend object
    for i,legobj in enumerate(legend.legendHandles):
      legobj.set_linewidth(4.0)


    mng = pl.get_current_fig_manager()
    # print(plotTrajParams['SubfigClustMaxWinSize'])
    # print(adsds)
    mng.resize(*windowSize)

    # pl.show()
    # print(adsa)

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    # import pdb
    # pdb.set_trace()


    #print("Plotting results .... ")
    pl.pause(0.05)

    return fig


  def plotTrajSamplesInOneNoData(self, dataCross, diagCross, dpsCross, subShiftsLong,
    ageFirstVisitLong1array, longDiag, uniquePartCodeInverse, crossAge1array, thetas, variances, clustProbBCColNorm, plotTrajParams,
     trajFunc, replaceFigMode=True, thetasSamples=None, showConfInt=True,
     colorTitle=True, yLimUseData=False,
     windowSize=(500,400), clustColsHues=None, normDPS=True):

    fontSize = plotTrajParams['TrajSamplesFontSize']
    adjustBottomHeight = plotTrajParams['TrajSamplesAdjBottomHeight']
    figSizeInch = (windowSize[0] / 100, windowSize[1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrClust = clustProbBCColNorm.shape[1]

    if normDPS:
      import voxelDPM
      subShiftsLongNorm, subShiftsCrossNorm, dpsLongNorm, dpsCrossNorm, thetasNorm, thetasSamplesNorm = \
        voxelDPM.VoxelDPM.normDPS(subShiftsLong, ageFirstVisitLong1array, longDiag, uniquePartCodeInverse,
          crossAge1array, thetas, thetasSamples)

      thetasSamples = thetasSamplesNorm
      thetas = thetasNorm
      dpsCross = dpsCrossNorm


    thetasSamplesClust = [ thetasSamples[c,:,:] for c in range(nrClust)]

    xs = np.linspace(np.min(dpsCross), np.max(dpsCross), 100)
    stdDevs = np.sqrt(variances)

    dpsSorted = np.sort(dpsCross)
    quantile = 0.98
    minX = dpsSorted[int(dpsSorted.shape[0]*(1-quantile))]
    maxX = dpsSorted[int(dpsSorted.shape[0]*quantile)]
    deltaX = (maxX - minX) / 10

    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']

    nrSubj, nrBiomk = dataCross.shape


    if clustColsHues is not None:
      clustOrderInd = np.argsort(clustColsHues)
    # elif orderClust:
    #   clustOrderInd = aux.orderTrajBySlope(thetas)
    else:
      clustOrderInd = range(nrClust)
      clustColsHues = np.linspace(0, 0.66, num=nrClust, endpoint=True)

    weightedDataMean = np.zeros((nrSubj, nrClust))

    for c in range(nrClust):
      # compute weighted mean of data
      weightedDataMean[:, c] = np.average(dataCross, axis=1, weights=clustProbBCColNorm[:, c])

    # color cluster traj according to the given colors
    clustColsRGB = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in clustColsHues]
    clustColsRGBperturb = [colorsys.hsv_to_rgb(hue, 0.3, 1) for hue in clustColsHues]

    if nrClust > 8:
      legIndToPlot = list(np.array(range(nrClust))[::int(nrClust/5)])
      if legIndToPlot[-1] != (nrClust-1):
        # add last cluster if it doesn't exist
        legIndToPlot = list(legIndToPlot) + [nrClust-1]
    else:
      legIndToPlot = range(nrClust)

    fsCurr = np.zeros((nrClust, xs.shape[0]), float)

    for c in range(nrClust):

      clustInd = clustOrderInd[c]
      if thetasSamplesClust is not None:
        for t in range(thetasSamplesClust[clustInd].shape[0]):
          fsSample = trajFunc(xs, thetasSamplesClust[clustInd][t, :])
          if plotTrajParams['biomkWasInversed']:
            fsSample *= -1
          pl.plot(xs, fsSample, color=clustColsRGBperturb[clustInd])

      fsCurr[c,:] = trajFunc(xs, thetas[clustInd, :])
      if plotTrajParams['biomkWasInversed']:
        fsCurr[c, :] *= -1

      # print(fsCurr.shape, xs.shape)
      if c in legIndToPlot:
        label = 'cluster %d' % c
      else:
        label = None
      pl.plot(xs, fsCurr[c,:], '-', linewidth=3.0, color=clustColsRGB[clustInd], label=label)  # label='sigmoid traj %d' % c

    pl.ylabel(plotTrajParams['biomkAxisLabel'], fontsize=fontSize)
    pl.xlabel('DPS', fontsize=fontSize)
    pl.tick_params(axis='both', which='major', labelsize=fontSize)
    pl.tick_params(axis='both', which='minor', labelsize=fontSize)

    pl.xlim(minX-deltaX, maxX+deltaX)
    # pl.ylim(-1.6,1)
    xsPlottedMask = np.logical_and(xs > minX-deltaX, xs < maxX + deltaX)

    sortedData = np.sort(weightedDataMean.reshape(-1))
    quantile = 0.99
    minY = np.min(fsCurr[:,xsPlottedMask],axis=(0,1))
    maxY = np.max(fsCurr[:,xsPlottedMask],axis=(0,1))
    delta = (maxY - minY) / 10

    if yLimUseData:
      yLim = [minY - delta, maxY + delta]
    else:
      yLim = plotTrajParams['ylimTrajSamplesInOneNoData']

    pl.ylim(yLim[0], yLim[1])

    # adjustCurrFig(plotTrajParams)
    pl.gcf().subplots_adjust(left=adjustBottomHeight, bottom=adjustBottomHeight)

    if plotTrajParams['trajSamplesPlotLegend']:
      legend = pl.legend()

      for i,legobj in enumerate(legend.legendHandles):
        legobj.set_linewidth(4.0)

      pl.setp(pl.gca().get_legend().get_texts(), fontsize=fontSize)


    mng = pl.get_current_fig_manager()
    mng.resize(*windowSize)

    if replaceFigMode:
      fig.show()
    else:
      pl.show()


    pl.pause(0.05)

    return fig

  def plotTrajWeightedDataMeanTrueParams(self, data, diag, dps, thetas, variances, clustProbBCColNorm,
     plotTrajParams, trajFunc, trueThetas, trueThetasPerturbedClust=None, replaceFigMode=True,
     showConfInt=True, colorTitle=True, adjustBottomHeight=0.25, fontsize=None):


    figSizeInch = (plotTrajParams['SubfigClustMaxWinSize'][0] / 100,
    plotTrajParams['SubfigClustMaxWinSize'][1] / 100)
    fig = pl.figure(figsize=figSizeInch)
    pl.clf()
    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']

    print(data.shape)
    nrSubj, nrBiomk = data.shape
    nrClust = clustProbBCColNorm.shape[1]

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj
    # dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)

    # dataSubset = data[dataSubsetIndices, :]
    # diagSubset = diag[dataSubsetIndices]
    # dpsSubset = dps[dataSubsetIndices]

    weightedDataMean = np.zeros((nrSubj, nrClust))

    for row in range(nrRows):
      for col in range(nrCols):
        c = row * nrCols + col # clusterNr
        if c < nrClust:
          ax = pl.subplot(nrRows, nrCols, c + 1)
          ax.set_title('cluster %s' % c)
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][c])

          if trueThetasPerturbedClust is not None:
            for t in range(trueThetasPerturbedClust[c].shape[0]):
              pl.plot(xs, trajFunc(xs, trueThetasPerturbedClust[c][t, :]), color='0.75', label='perturbed trajectories')

          fsCurr = trajFunc(xs, thetas[c,:])
          #print(fsCurr.shape, xs.shape)
          pl.plot(xs, fsCurr, 'b-', linewidth=3.0, label='fitted trajectory')
          if showConfInt:
            pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[c],
              (fsCurr + 1.9600 * stdDevs[c])[::-1]]), alpha=.3, fc='b', ec='None')
                    #label='conf interval (1.96*std)')

          fsCurrTrue = trajFunc(xs, trueThetas[c, :])
          pl.plot(xs, fsCurrTrue, 'r-', linewidth=3.0, label='true trajectory')


          #print(xs[50:60], fsCurr[50:60], thetas[c,:])
          #print(asda)

          # compute weighted mean of data
          weightedDataMean[:,c] = np.average(data, axis=1,weights=clustProbBCColNorm[:,c])

          #lines = []
          # for d in range(len(diagNrs)):
          #   #print(clustSubsetIndices, data[diag == diagNrs[d], clustSubsetIndices[c]].shape)
          #   pl.scatter(dpsSubset[diagSubset == diagNrs[d]],
          # weightedDataMean[diagSubset == diagNrs[d],c], s=20,
          # c=plotTrajParams['diagColors'][diagNrs[d]],
          #     label=plotTrajParams['diagLabels'][diagNrs[d]])

          if col == 0:
            pl.ylabel('vertex value')
          if row == nrRows - 1:
            pl.xlabel('DPS')

          if col > 0:
            ax.set_yticklabels([])

          pl.xlim(np.min(dps), np.max(dps))
          pl.ylim(np.min(weightedDataMean),
                  np.max(weightedDataMean))
          # pl.ylim(-4,0.2)

    adjustCurrFig(plotTrajParams)

    pl.gcf().subplots_adjust(bottom=adjustBottomHeight)
    #fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    #print(h[2:4], labels[2:4])
    #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    # print(len(axisLabels), h)

    legend = pl.figlegend(h[-3:], axisLabels[-3:], loc='lower center', ncol=3, labelspacing=0. )
    # set the linewidth of each legend object
    for i,legobj in enumerate(legend.legendHandles):
      legobj.set_linewidth(4.0)
      #legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

    mng = pl.get_current_fig_manager()
    mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    if fontsize:
      # SIZE = fontsize
      # pl.rc('font', size = SIZE)  # controls default text sizes
      # pl.rc('axes', titlesize = SIZE)  # fontsize of the axes title
      # pl.rc('axes', labelsize = SIZE)  # fontsize of the x and y labels
      # pl.rc('xtick', labelsize = SIZE)  # fontsize of the tick labels
      # pl.rc('ytick', labelsize = SIZE)  # fontsize of the tick labels
      # pl.rc('legend', fontsize = SIZE)  # legend fontsize

      # pl.tight_layout()
      pl.subplots_adjust(left = None, bottom = 0.25, right = None, top = None, wspace = 0.05, hspace = None)


    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    #print("Plotting results .... ")
    pl.pause(0.05)

    return fig

  def plotSubShiftsTrue(self, subShifts, subShiftsTrue, dpsLong, dpsLongTrue, plotTrajParams, replaceFigMode=True,
    fontsize=None):


    figSizeInch = (plotTrajParams['SubfigClustMaxWinSize'][1] / 100,
    plotTrajParams['SubfigClustMaxWinSize'][1] / 100)
    fig = pl.figure(figsize=figSizeInch)
    pl.clf()

    if fontsize:
      SIZE = fontsize
      pl.subplots_adjust(left=0.2, bottom=0.15, right=None, top=None, wspace=None, hspace=None)

      font = {'family': 'normal',
        'weight': 'bold',
        'size': fontsize}

      # matplotlib.rc('font', **font)

      matplotlib.rcParams.update({'font.size': fontsize})

      # params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'text.fontsize': fontsize, 'legend.fontsize': fontsize,
      #   'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
      # matplotlib.rcParams.update(params)


    # ax = pl.subplot(1, 3, 1)
    # pl.plot(subShifts[:,0], 'k-', label='estimated alphas')
    # pl.plot(subShiftsTrue[:,0], 'r-', label='true alphas') # alpha
    # pl.xlabel("Subject Nr.")
    # pl.ylabel("Alpha")
    #
    # ax = pl.subplot(1, 3, 2)
    # pl.plot(subShifts[:, 1], 'k-', label='estimated betas')
    # pl.plot(subShiftsTrue[:, 1], 'r-', label='true betas')  # beta
    # pl.xlabel("Subject Nr.")
    # pl.ylabel("Beta")

    mng = pl.get_current_fig_manager()
    # mng.resize(*(300,400))

    # ax = pl.subplot(1, 3, 3)
    pl.scatter(dpsLongTrue, dpsLong)
    pl.xlabel("true DPS")
    pl.ylabel("estimated DPS")
    axes = pl.gca()
    xMin, xMax = axes.get_xlim()
    yMin, yMax = axes.get_ylim()

    min = np.min([xMin, yMin])
    max = np.max([xMax, yMax])

    pl.plot([-100, 100], [-100, 100])
    axes.set_xlim([min, max])
    axes.set_ylim([min, max])

    # print(adsa)

    # pl.tick_params(axis='both', which='major', labelsize=14)

    # h, axisLabels = pl.gca().get_legend_handles_labels()
    # legend = pl.figlegend(h, axisLabels, loc='upper center', ncol=2, labelspacing=0.)
    # for i,legobj in enumerate(legend.legendHandles):
    #   legobj.set_linewidth(4.0)

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    return fig

  def plotClustProb(self, clustProbBC, thetas, variances, subShiftsLong,
    plotTrajParams, filePathNoExt):
    """save the clustProb to a file and then generates a 3D blender rendering"""
    filePath = '%s.npz' % filePathNoExt
    dataStruct = dict(clustProbBC=clustProbBC, thetas=thetas, variances=variances,
      subShiftsLong=subShiftsLong, plotTrajParams=plotTrajParams)
    pickle.dump(dataStruct, open(filePath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    cmd = 'file=%s pngFile=%s.png isCluster=%d %s --background --python ' \
          'colorClustProb.py' % (filePath, filePathNoExt,
    plotTrajParams['cluster'], plotTrajParams['blenderPath'])
    print(cmd)
    os.system(cmd)

    if plotTrajParams['reduceSpace']:
      os.system('rm %s' % filePath)


  # def plotClustProb(self, clustProbBC, thetas, variances, subShiftsLong,
  #   plotTrajParams, filePathNoExt):
  #   """save the clustProb to a file and then generates a 3D blender rendering"""




  def plotDiffs(self, diffsB, plotTrajParams, filePathNoExt):
    """save the clustProb to a file and then generates a 3D blender rendering"""
    filePath = '%s.npz' % filePathNoExt
    dataStruct = dict(diffsB=diffsB, plotTrajParams=plotTrajParams)
    pickle.dump(dataStruct, open(filePath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    cmd = 'file=%s pngFile=%s.png isCluster=%d %s --background --python ' \
          'colorDiffs.py' % (filePath, filePathNoExt,
    plotTrajParams['cluster'], plotTrajParams['blenderPath'])
    print(cmd)
    os.system(cmd)

    if plotTrajParams['reduceSpace']:
      os.system('rm %s' % filePath)


  def colorClustByAtrophyExtent(self, thetas, dpsCross,
    diagCross, patientID, trajFunc):
    nrClust = thetas.shape[0]
    # dpsPatMean = np.mean(dpsCross[diagCross == patientID])
    # dpsPatStd = np.mean(dpsCross[diagCross == patientID])
    # dpsCtlMean = np.mean(dpsCross[diagCross == CTL])
    # dpsCtlStd = np.mean(dpsCross[diagCross == CTL])
    # dpsThresh = dpsCtlMean + 3*dpsCtlStd
    # dpsThresh = dpsPatMean + 1 * dpsPatStd
    dpsThresh = 3
    print('dpsThresh', dpsThresh)
    biomkValuesThresh = np.zeros(nrClust, float)
    for c in range(nrClust):
      biomkValuesThresh[c] = trajFunc(dpsThresh, thetas[c,:])

    print('biomkValuesThresh', biomkValuesThresh)

    minHue = 0 # min Hue is 0=red
    maxHue = 0.66 # max Hue is 1, which is the same as 0. recommended 0.66=blue
    # make the colors uniformly distanced by sorting twice
    hueValsC = np.argsort(np.argsort(biomkValuesThresh))
    # scale the numbers, make them between (0,1)
    hueValsC = (hueValsC - np.min(hueValsC))/\
              (np.max(hueValsC) - np.min(hueValsC))
    # scale the numbers again, make them between (minHue,maxHue)
    hueValsC = minHue + hueValsC*(maxHue-minHue)

    rgbValsC = [ matplotlib.colors.hsv_to_rgb([hue, 1, 1]) for hue in hueValsC]

    sortedInd = np.argsort(hueValsC)

    print('hueValsC', hueValsC)
    print('rgbValsC', rgbValsC)
    print('sortedInd', sortedInd)
    # print(adsa)

    return hueValsC, rgbValsC, sortedInd

  def colorClustByAtrophyExtentAtDps(self, thetas, dpsThresh, minBiomkVal, maxBiomkVal, trajFunc):

    nrClust = thetas.shape[0]
    print('dpsThresh', dpsThresh)
    biomkValuesThresh = np.zeros(nrClust, float)
    for c in range(nrClust):
      biomkValuesThresh[c] = trajFunc(dpsThresh, thetas[c,:])

    print('biomkValuesThresh', biomkValuesThresh)

    minHue = 0 # min Hue is 0=red
    maxHue = 0.66 # max Hue is 1, which is the same as 0. recommended 0.66=blue
    # make the colors uniformly distanced by sorting twice
    # hueValsC = np.argsort(np.argsort(biomkValuesThresh))

    hueValsC = biomkValuesThresh
    # scale the numbers, make them between (0,1)
    # hueValsC = (hueValsC - np.min(hueValsC))/\
    #           (np.max(hueValsC) - np.min(hueValsC))
    hueValsC = (hueValsC - minBiomkVal) / \
               (maxBiomkVal - minBiomkVal)

    # scale the numbers again, make them between (minHue,maxHue)
    hueValsC = minHue + hueValsC*(maxHue-minHue)

    rgbValsC = [ matplotlib.colors.hsv_to_rgb([hue, 1, 1]) for hue in hueValsC]

    sortedInd = np.argsort(hueValsC)

    print('hueValsC', hueValsC)
    print('rgbValsC', rgbValsC)
    print('sortedInd', sortedInd)
    # print(adsa)

    return hueValsC, rgbValsC, sortedInd

  def colorClustBySlope(self, thetas):
    ''' Generates colors for the clusters!'''
    minHue = 0
    maxHue = 0.66
    slopes = (thetas[:, 0] * thetas[:, 1]) / 4
    nrClust = thetas.shape[0]
    slopesSortedInd = np.argsort(slopes)

    minSlope = np.min(slopes)
    maxSlope = np.max(slopes)
    hueValsC = (slopes - minSlope) / (maxSlope - minSlope) # scale to (0,1)
    hueValsC = minHue + hueValsC * maxHue # scale to (minHue, maxHue)

    rgbValsC3 = np.array([colorsys.hsv_to_rgb(hue, 1, 1) for hue in
      hueValsC]) # shape is Cx3
    assert rgbValsC3.shape[1] == 3

    return hueValsC, rgbValsC3, slopesSortedInd

  def plotBlenderDirectGivenClustCols(self, hueColsC, clustProbBC, plotTrajParams, filePathNoExt):
    """generates a 3D blender brain where regions are colored according
    to the atrophy extent at a particular threshold"""

    # compute the hues for each vertex by interpolating the cols using the  clustering probabilities
    vertexColsB = np.dot(clustProbBC, hueColsC)
    # print('vertexColsB', vertexColsB[::1000])
    vertexColsHue = vertexColsB[plotTrajParams['nearestNeighbours']]
    # import pdb
    # pdb.set_trace()
    vertexCols = matplotlib.colors.hsv_to_rgb(np.concatenate((vertexColsHue.reshape(-1,1),
      np.ones(vertexColsHue.shape).reshape(-1,1),\
      np.ones(vertexColsHue.shape).reshape(-1,1)), axis=1))

    # print('vertexCols', vertexCols.shape, vertexCols[::1000, :])

    dataStruct = dict(vertexCols=vertexCols)
    filePathNpz = '%s.npz' % filePathNoExt
    pickle.dump(dataStruct, open(filePathNpz, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    cmd = 'file=%s pngFile=%s.png isCluster=%d %s --background --python ' \
          'colorVerticesDirectly.py' % (filePathNpz, filePathNoExt,
    plotTrajParams['cluster'], plotTrajParams['blenderPath'])
    print(cmd)
    os.system(cmd)

    if plotTrajParams['reduceSpace']:
      os.system('rm %s' % filePathNpz)

  def makeMovie(self, thetas, dpsCross, crossDiag, trajFunc, clustProb, plotTrajParams,
            filePathNoExt):

    dpsSorted = np.sort(dpsCross)
    quantile = 0.95

    nrThresh = 21
    # minX = np.min(dpsCross)
    minX = np.mean(dpsCross[crossDiag == CTL])

    # maxX = np.max(dpsCross)
    maxX = dpsSorted[int(dpsSorted.shape[0]*quantile)]
    threshList = np.linspace(minX, maxX , num=nrThresh)

    print('dpsCross', dpsCross)
    print('threshList', threshList)

    nrClust = thetas.shape[0]
    minC = np.zeros(nrClust)
    maxC = np.zeros(nrClust)
    for c in range(nrClust):
      biomkValues = trajFunc(threshList, thetas[c,:])
      print('biomkValues', biomkValues)
      minC[c] = np.min(biomkValues)
      maxC[c] = np.max(biomkValues)

    maxBiomkVal = np.max(maxC)
    minBiomkVal = np.min(minC)

    print('minBiomkVal', minBiomkVal)
    print('maxBiomkVal', maxBiomkVal)
    # print(ads)

    for t in range(nrThresh):

      hueColsAtrophyExtentC, rgbColsAtrophyExtentC, sortedIndAtrophyExtent = \
        self.colorClustByAtrophyExtentAtDps(thetas, threshList[t], minBiomkVal, maxBiomkVal, trajFunc)

      tFileName = t + 10
      filePathNoExtCurrThresh = '%s_%d' % (filePathNoExt, tFileName)
      self.plotBlenderDirectGivenClustCols(hueColsAtrophyExtentC, clustProb, plotTrajParams,
                                      filePathNoExt=filePathNoExtCurrThresh)
      cwd = os.getcwd()
      textCmd = 'cd %s; convert -pointsize 30 -fill black -draw \'text 0,30 "DPS %.1f (%2d%%)" \' %s_%d.png %s_%d.png' \
                % (cwd, threshList[t], float(100*t)/(nrThresh-1), filePathNoExt, tFileName, filePathNoExt + 'text', tFileName)
      print(textCmd)
      os.system(textCmd)

      # print(adsa)

  def makeSnapshots(self, thetas, dpsCross, crossDiag, trajFunc, clustProb, plotTrajParams,
            filePathNoExt):
    """
    Makes snapshots at DPS=1 and DPS=2
    """

    dpsSorted = np.sort(dpsCross)
    quantile = 0.95


    # minX = np.min(dpsCross)
    minX = np.mean(dpsCross[crossDiag == CTL])

    # maxX = np.max(dpsCross)
    maxX = dpsSorted[int(dpsSorted.shape[0]*quantile)]
    threshList = np.array([0.25, 0.5, 1, 2, 4], int)
    nrThresh = threshList.shape[0]

    print('dpsCross', dpsCross)
    print('threshList', threshList)

    nrClust = thetas.shape[0]
    minC = np.zeros(nrClust)
    maxC = np.zeros(nrClust)
    for c in range(nrClust):
      biomkValues = trajFunc(threshList, thetas[c,:])
      print('biomkValues', biomkValues)
      minC[c] = np.min(biomkValues)
      maxC[c] = np.max(biomkValues)

    maxBiomkVal = np.max(maxC)
    minBiomkVal = np.min(minC)

    print('minBiomkVal', minBiomkVal)
    print('maxBiomkVal', maxBiomkVal)

    # print(asd)
    # print(ads)

    for t in range(nrThresh):

      hueColsAtrophyExtentC, rgbColsAtrophyExtentC, sortedIndAtrophyExtent = \
        self.colorClustByAtrophyExtentAtDps(thetas, threshList[t], minBiomkVal, maxBiomkVal, trajFunc)


      self.plotBlenderDirectGivenClustCols(hueColsAtrophyExtentC, clustProb, plotTrajParams,
                                      filePathNoExt='%s_%d' % (filePathNoExt, t) )



  def writeAnnotFile(self, clustProbBC, thetas, variances, subShiftsLong,
    plotTrajParams, filePathNoExt):
    """
    generated annotation file, creates one label group for each vertex, quite inneficient ...
    :param clustProbBC:
    :param plotTrajParams:
    :param filePathNoExt: filepath without extension
    :return:
    """
    nrBiomk, nrClust = clustProbBC.shape
    if nrClust < 36:
      pointIndices = plotTrajParams['pointIndices']
      nearestNeighbours = plotTrajParams['nearestNeighbours']
      fsaverageInflatedLh = '%s/subjects/fsaverage/surf/lh.inflated' % \
                            plotTrajParams['freesurfPath']
      coordsLh, facesLh, _ = nib.freesurfer.io.read_geometry(fsaverageInflatedLh, read_metadata=True)

      fsaverageInflatedRh = '%s/subjects/fsaverage/surf/rh.inflated' % \
                            plotTrajParams['freesurfPath']
      coordsRh, facesRh, _ = nib.freesurfer.io.read_geometry(fsaverageInflatedRh, read_metadata=True)

      # print(coordsLh.shape, coordsLh[1:30,:])

      print(np.sum(clustProbBC, 1))
      assert (all(np.abs(np.sum(clustProbBC, 1) - 1) < 0.001))

      clustHuePoints = plotTrajParams['clustHuePoints']
      print(clustHuePoints)

      colsB = np.zeros((nrBiomk, 3), int)
      colsC = np.zeros((nrClust, 3), int)
      bestClust = np.zeros(nrBiomk, int)

      for b in range(nrBiomk):  # nr points analysed, 10k for the small dataset but filtered for zero biomk
        bestClust[b] = np.argmax(clustProbBC[b, :])
        hue = clustHuePoints[bestClust[b]]
        colsB[b, :] = [int(255 * i) for i in list(colorsys.hsv_to_rgb(hue, 1, 1))]

      for c in range(nrClust):
        hue = clustHuePoints[c]
        colsC[c, :] += [int(255 * i) for i in list(colorsys.hsv_to_rgb(hue, 1, 1))]

      nrVertices = coordsLh.shape[0]
      defLabel = nrClust
      # labels = defLabel * np.ones(nrVertices, int)
      # labels[pointIndices] = bestClust
      labels = bestClust[nearestNeighbours]
      assert (labels.shape[0] == nrVertices)

      nrColors = nrClust

      ctab = 128 * np.ones((nrColors, 5), int)
      ctab[0:nrClust, :][:, 0:3] = colsC
      ctab[:, 3] = 255
      ctab[:, 4] = range(1, nrClust + 1)
      names = ['clust%d' % i for i in range(nrClust)] + ['point not analysed']

      print(ctab)

      fsaverageAnnotFile = '%s/subjects/fsaverage/label/lh.aparc.annot' % \
                           plotTrajParams['freesurfPath']
      labels2, ctab2, _ = nib.freesurfer.io.read_annot(fsaverageAnnotFile, orig_ids=False)
      # labels2 = np.ones(nrVertices, int)
      print(labels2[1:100])
      for v in range(nrVertices):
        labels2[v] = nrClust - 1

      labels2[pointIndices] = bestClust

      print(np.unique(labels2))
      print(np.unique(labels))
      assert(len(np.unique(labels2)) == len(np.unique(labels)))

      print('labels[:100]', labels[:100])
      print('ctab2[0:(nrClust+1),:]', ctab2[0:nrColors, :])
      print('names[0:(nrClust+1)]', names[0:nrColors])

      # print(np.min(labels2), np.max(labels2), labels2[1:1000])
      # print(labels2.shape, labels.shape)
      # print(type(ctab), type(ctab2))
      # print(ctab.shape, ctab2[0:5,:].shape)
      # print([coordsLh[facesLh[i,:],:] for i in range(5)])
      # print([coordsLh[i, :] for i in range(5)])
      # print([facesLh[i, :] for i in range(5)])

      # ctab[:,4] = ctab2[0:(nrClust+1),4]
      # ctab2[0:(nrClust+1),0:3] = ctab[:,0:3]
      # print(ctab, ctab2[0:5,:])

      # nib.freesurfer.io.write_annot('%s.annot' % filePathNoExt,
      #   labels, ctab2[1:(nrColors+1), :], names[1:(nrColors+1)])
      nib.freesurfer.io.write_annot('%s.annot' % filePathNoExt,
        labels, ctab2[1:(nrColors+1), :], names[:nrColors])

      # print(adsa)

  def makeSnapshotBlender(self, filePath, plotTrajParams):

    cmd = 'file=%s isCluster=%d %s --background --python colorClustProb.py' % (filePath,
    plotTrajParams['cluster'], plotTrajParams['blenderPath'])
    print(cmd)
    os.system(cmd)

  def colSlopeBlender(self, filePath, pngfile, plotTrajParams):

    cmd = 'file=%s  pngFile=%s isCluster=%d %s --background --python colorClustSlope.py' % (filePath,
    pngfile, plotTrajParams['cluster'], plotTrajParams['blenderPath'])
    print(cmd)
    os.system(cmd)


########## End of Base class PlotterVDPM ##############

class PlotterVDPMSynthMrf(PlotterVDPM):

  ''' standard class for generating a colouring of the cluters in a synthetic experiment'''

  def __init__(self):
    super().__init__()

  def plotClustProb(self, clustProbBC, thetas, variances, subShiftsLong,
    plotTrajParams, filePathNoExt):
    """
    generates a plot of points on a 2D lattice representing vertices,
    colored by the most likely cluster they belong

    :param clustProbBC:
    :param plotTrajParams:
    :param filePathNoExt: filepath without extension
    :return:
    """
    nrBiomk, nrClust = clustProbBC.shape

    nrRows = int(np.sqrt(nrBiomk))
    nrCols = nrRows

    xs = np.linspace(0, 1, nrRows)
    ys = np.linspace(0, 1, nrCols)
    xx, yy = np.meshgrid(xs, ys)

    xxLin = xx.reshape((1, -1))
    yyLin = yy.reshape((1, -1))

    clustMax = np.array(np.argmax(clustProbBC, axis=1), int)
    # print('clustMax', clustMax)
    fig = pl.figure(2)
    pl.clf()

    pl.scatter(xxLin, yyLin, c=[plotTrajParams['clustCols'][i] for i in clustMax],
      s=5)

    fig.show()
    fig.savefig('%s.png' % filePathNoExt, dpi=100)

    return fig

class PlotterVDPMSynth(PlotterVDPM):
  def __init__(self):
    super().__init__()

  def plotBlenderDirectGivenClustCols(self, hueColsC, clustProbBC, plotTrajParams, filePathNoExt):

    pass

  def plotClustProb(self, clustProbBC, thetas, variances, subShiftsLong,
    plotTrajParams, filePathNoExt):
    """
    doesn't generate any image as it's synthetic data experiment
    :param clustProbBC:
    :param plotTrajParams:
    :param filePathNoExt: filepath without extension
    :return:
    """
    pass

  def plotSynthResOneExp(self, numbers, stds, xTickValues, xLabelStr, yLabelStr, makeInts=False, adjLeft=0.15):
    """

    :param resList:
    :param xTickValues:
    :param xLabelStr:
    :return:
    """

    fig = pl.figure(1)
    pl.clf()

    fs = 18

    twoSigmaRange = 2*np.sqrt((np.array(stds) ** 2)/stds.shape[0]) # compute the std deviation of the sample mean by classical formula

    # twoSigmaRange = 2*stds

    print(twoSigmaRange.shape)

    pl.errorbar(range(len(numbers)), numbers, yerr=twoSigmaRange, linewidth=3)
    ax = pl.gca()
    print(np.floor(numbers[0]), numbers[0] )
    allIntegers = np.array([np.floor(i) == i for i in numbers]).all()
    if makeInts:
      ax.set_xticklabels([''] + ['%d' % int(i) for i in xTickValues], fontsize=fs)
    else:
      ax.set_xticklabels([''] + ['%.1f' % i for i in xTickValues], fontsize=fs)
    pl.yticks(fontsize=fs)
    pl.xticks(fontsize=fs)
    # pl.ylim([0,1])

    pl.xlabel(xLabelStr, fontsize=fs)
    pl.ylabel(yLabelStr, fontsize=fs)

    pl.gcf().subplots_adjust(bottom=0.15,left=adjLeft)

    fig.show()
    return fig

  def boxPlotResOneExp(self, numbers, xTickValues, xLabelStr, yLabelStr, makeInts=False, adjLeft=0.18, xscale=None,
                       yscale=None, bootstrap=False):
    """

    :param resList:
    :param xTickValues:
    :param xLabelStr:
    :return:
    """

    fig = pl.figure(3)
    pl.clf()

    # fs = 24
    #
    # fd = {'family': 'DejVu Sans',
    #   'color': 'black',
    #   'fontweight': 'heavy',
    #   'weight': 'heavy',
    #   'size': 24,
    # }

    if bootstrap:
      nrSteps = len(numbers)
      nrBs = 10
      numbersBs = [np.zeros(nrBs) for _ in range(nrSteps)] # numbers from bootstrap
      # import numpy.random
      for s in range(nrSteps):
        for b in range(nrBs):
          print(type(numbers[s]), numbers[s].shape)
          numbersBs[s][b] = np.mean(np.random.choice(numbers[s], size=numbers[s].shape[0], replace=True))

      numbers = numbersBs

    # pl.errorbar(range(len(numbers)), numbers, yerr=twoSigmaRange, linewidth=3)
    pl.boxplot(numbers, showfliers=True, whis=10, notch=False, showmeans=True)

    ax = pl.gca()
    if yscale is not None:
      ax.set_yscale(yscale)

    # if xscale is not None:
    #   ax.set_xscale(xscale)

    # print(np.floor(numbers[0]), numbers[0] )
    # allIntegers = np.array([np.floor(i) == i for i in numbers]).all()
    if makeInts:
      ax.set_xticklabels(['%d' % int(i) for i in xTickValues])
    else:
      ax.set_xticklabels(['%.2f' % i for i in xTickValues])

    # ax.set_yticklabels(['\\textbf{%s}' % i for i in ax.get_yticklabels()])

    pl.yticks()
    pl.xticks(xTickValues)
    #pl.xticks()
    # pl.ylim([0,1])

    pl.xlabel('%s' % xLabelStr)
    pl.ylabel('%s' % yLabelStr)

    pl.gcf().subplots_adjust(bottom=0.15,left=adjLeft)

    fig.show()
    return fig

  def scatterPlotNrClust(self, trueNrOfClust, estClustAIC, estClustBIC, xLabelStr, yLabelStr):


    fig = pl.figure(8)
    pl.clf()

    # pl.rc('text', usetex=True)
    #
    # pgf_with_latex = {  # setup matplotlib to use latex for output# {{{
    #   "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    #   "text.usetex": True,  # use LaTeX to write all text
    #   "font.family": "DejaVu Sans",
    #   "font.serif": [],  # blank entries should cause plots
    #   "font.sans-serif": [],  # to inherit fonts from the document
    #   "font.monospace": [],
    #   "axes.labelsize": 12,  # LaTeX default is 10pt font.
    #   "font.size": 12,
    #   "legend.fontsize": 12,  # Make the legend/label fonts
    #   "xtick.labelsize": 12,  # a little smaller
    #   "ytick.labelsize": 12,
    #   "pgf.preamble": [
    #     r"\usepackage{amssymb}",
    #     r'\boldmath'
    #   ]  # using this preamble
    # }
    # # }}}
    # pl.rcParams.update(pgf_with_latex)

    fig = pl.figure(1)
    pl.clf()

    # fs = 18
    #
    # fd = {'family': 'serif',
    #   'color': 'black',
    #   'weight': 'normal',
    #   'size': 18,
    # }

    # fs = 18

    pl.scatter(np.log(trueNrOfClust), np.log(estClustAIC), s=80, marker='o', label='AIC', c='b')
    pl.scatter(np.log(trueNrOfClust), np.log(estClustBIC), s=80, marker='x', label='BIC', c='r')

    pl.xlabel(xLabelStr)
    pl.ylabel(yLabelStr)

    axes = pl.gca()
    xMin, xMax = axes.get_xlim()
    yMin, yMax = axes.get_ylim()

    min = np.min([xMin, yMin])
    max = np.max([xMax, yMax])

    pl.plot([-100, 100], [-100, 100])
    axes.set_xlim([min, max])
    axes.set_ylim([min, max])

    axes.set_xticks(np.log(trueNrOfClust))
    axes.set_yticks(np.log(trueNrOfClust))

    # axes.set_yscale('log')
    # axes.set_xscale('log')

    axes.set_xticklabels(['%d' % i for i in trueNrOfClust])
    axes.set_yticklabels(['%d' % i for i in trueNrOfClust])

    pl.gcf().subplots_adjust(bottom=0.15)

    pl.legend()

    fig.show()
    return fig


class PlotterVDPMScalar(PlotterVDPM):

  ''' class used for plotting a dataset of scalar biomarkers
  (e.g. ROI measures from MRI, PET, DTI, can be all combined)

  also works when the dataset contains NaN values

  '''

  def __init__(self):
    super().__init__()
    self.nanMask = None
    self.longDataNaNs = None

  def plotClustProb(self, clustProbBC, thetas, variances, subShiftsLong,
    plotTrajParams, filePathNoExt):
    """
    doesn't generate any image as it's synthetic data experiment

    plotClustProb could potentially plot a dendogram showing how measures are grouped into clusters
    ,but it is not implemented yet

    :param clustProbBC:
    :param plotTrajParams:
    :param filePathNoExt: filepath without extension
    :return:
    """
    pass


  def plotTrajWeightedDataMean(self, data, diag, dps, longData, longDiag, longDPS, thetas, variances,
    clustProbBCColNorm, plotTrajParams, trajFunc, replaceFigMode=True, thetasSamplesClust=None,
    showConfInt=True, colorTitle=False, yLimUseData=False, adjustBottomHeight=0.15,
    orderClust=False, showInferredData=False, varyingAlpha=True):

    figSizeInch = (plotTrajParams['SubfigClustMaxWinSize'][0]/100,
    plotTrajParams['SubfigClustMaxWinSize'][1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrRows = plotTrajParams['nrRows']
    nrCols = plotTrajParams['nrCols']


    nrSubj, nrBiomk = data.shape
    nrClust = clustProbBCColNorm.shape[1]

    # if orderClust:
    #   clustOrderInd = aux.orderTrajBySlope(thetas)
    # else:
    #   clustOrderInd = range(nrClust)

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj

    nrSubjLong = len(longData)
    # weightedDataMeanSCInfer = np.zeros((nrSubj, nrClust))
    # weightedDataMeanLongInfer = [np.zeros((longData[s].shape[0], nrClust),float) for s in range(nrSubjLong)]

    # for c in range(nrClust):
    #   # compute weighted mean of data
    #   weightedDataMeanSCInfer[:, c] = np.average(data, axis=1, weights=clustProbBCColNorm[:, c])
    #
    #   for s in range(nrSubjLong):
    #     weightedDataMeanLongInfer[s][:,c] = np.average(longData[s], axis=1, weights=clustProbBCColNorm[:, c])

    weightedDataMeanSCNoInfer = ma.zeros((nrSubj, nrClust))
    weightedDataMeanLongNoInfer = [ma.zeros((longData[s].shape[0], nrClust),float) for s in range(nrSubjLong)]
    longAlphaLevel = [0 for s in range(nrSubjLong) ]
    crossAlphaLevelSC = np.zeros((nrSubj, nrClust), float)

    # print(np.sum(clustProbBCColNorm[:,0]))
    # print(adsa)

    for c in range(nrClust):
      # compute NaN-ignoring weighted mean of data
      weightedDataMeanSCNoInfer[:, c] = np.ma.average(data, axis=1, weights=clustProbBCColNorm[:, c])
      dataNNMask = np.logical_not(data.mask)
      crossAlphaLevelSC[:,c] = np.sum(dataNNMask * clustProbBCColNorm[:, c][None,:],axis=1)

      for s in range(nrSubjLong):
        weightedDataMeanLongNoInfer[s][:,c] = np.ma.average(longData[s], axis=1,
          weights=clustProbBCColNorm[:, c])

        # dataNNMask = np.logical_not(longData[s].mask)
        # longAlphaLevel[s] = np.sum(dataNNMask * clustProbBCColNorm[:, c][None, :], axis=1)

    if showInferredData:
      weightedDataMeanSC = weightedDataMeanSCInfer
      weightedDataMeanLong = weightedDataMeanLongInfer
    else:
      weightedDataMeanSC = weightedDataMeanSCNoInfer
      weightedDataMeanLong = weightedDataMeanLongNoInfer

    # print('weightedDataMeanSCNoInfer', weightedDataMeanSCNoInfer[0, :])
    # print('longData[0]', longData[0])
    #
    # print('weightedDataMeanLongNoInfer', weightedDataMeanLongNoInfer[0])
    # # print('-----------------------')
    # print(ads)

    # subNr = 0
      # clustNr = 0
      # notNanMaskSub0B = np.logical_not(np.isnan(dataNaNs[subNr,:]))
      # weightsNanNorm = clustProbBCColNorm[:, clustNr] / np.sum(clustProbBCColNorm[notNanMaskSub0B, clustNr])
      # weightedDataMeanSub0Cl0 = np.nansum(weightsNanNorm * dataNaNs[subNr,:])
      #
      # print('weightedDataMeanSub0Cl0', weightedDataMeanSub0Cl0)
      # print('weightedDataMeanSC[subNr, clustNr]', weightedDataMeanSC[subNr, clustNr])
      # print('weightsNanNorm', weightsNanNorm)
      # print('dataNaNs[subNr,:]', dataNaNs[subNr,:])
      # assert abs(weightedDataMeanSub0Cl0 - weightedDataMeanSC[subNr, clustNr]) < 0.001
      #
      # print(adsa)

    limWeightedDataMeanSC = weightedDataMeanSC.copy()
    limWeightedDataMeanSC[crossAlphaLevelSC < 0.5] = np.nan

    yMinC = np.nanmin(limWeightedDataMeanSC, axis=0)
    yMaxC = np.nanmax(limWeightedDataMeanSC, axis=0)
    delta = (yMaxC - yMinC) / 10

    fsCurrCT = np.zeros((nrClust, xs.shape[0]), np.float)

    # print('nrRows', nrRows)
    # print('nrCols', nrCols)
    # print(adsa)

    for row in range(nrRows):
      for col in range(nrCols):
        c = row * nrCols + col # clusterNr
        if c < nrClust:
          ax = pl.subplot(nrRows, nrCols, c + 1)
          sortedBiomkIndInCurrClust = np.argsort(clustProbBCColNorm[:, c])[::-1]
          # print('clustProbBCColNorm', clustProbBCColNorm)
          # print(ads)
          # print('sortedBiomkIndInCurrClust', sortedBiomkIndInCurrClust)
          ax.set_title('c%s %s' % (c,
            plotTrajParams['labels'][sortedBiomkIndInCurrClust[0]][:10]))
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][c])

          fsCurrCT[c,:] = trajFunc(xs, thetas[c,:])

          pl.plot(xs, fsCurrCT[c,:], 'k-', linewidth=3.0) # label='sigmoid traj %d' % c
          if showConfInt:
            pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurrCT[c,:] - 1.9600 * stdDevs[c],
              (fsCurrCT[c,:] + 1.9600 * stdDevs[c])[::-1]]), alpha=.3, fc='b', ec='None')
                    #label='conf interval (1.96*std)')

          if thetasSamplesClust is not None:
            for t in range(thetasSamplesClust[c].shape[0]):
              pl.plot(xs, trajFunc(xs, thetasSamplesClust[c][t, :]))

          ############# scatter plot subjects ######################

          if varyingAlpha:
            nrLevels = 11
            alphaLevels = np.linspace(0,1, nrLevels, True)
            deltaAlpha = 1.0/(nrLevels-1)
            for a in range(1,nrLevels):
              # print('crossAlphaLevelSC[:,c]', crossAlphaLevelSC[:,c])
              dataInd = np.logical_and(crossAlphaLevelSC[:,c] <= alphaLevels[a],
                                       crossAlphaLevelSC[:,c] >= (alphaLevels[a]-deltaAlpha))
              # print('dataInd', dataInd)
              for d in range(len(diagNrs)):
                finalMask = np.logical_and(dataInd, diag == diagNrs[d])
                # print('finalMask', finalMask)
                # print(adsa)
                if a == 1:
                  labelCurr = plotTrajParams['diagLabels'][diagNrs[d]]
                else:
                  labelCurr = ''
                pl.scatter(dps[finalMask], weightedDataMeanSC[finalMask, c],
                  s=20, c=plotTrajParams['diagColors'][diagNrs[d]],
                  label=labelCurr, alpha=alphaLevels[a])

          else:
            for d in range(len(diagNrs)):
              #print(clustSubsetIndices, dataSubset[diag == diagNrs[d], clustSubsetIndices[c]].shape)
              pl.scatter(dps[diag == diagNrs[d]], weightedDataMeanSC[diag == diagNrs[d],c],
              s=20, c=plotTrajParams['diagColors'][diagNrs[d]],
                label=plotTrajParams['diagLabels'][diagNrs[d]]
              )

          ############# spagetti plot subjects ######################

          # counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
          # for s in range(nrSubjLong):
          #   #print(clustSubsetIndices, dataSubset[diag == diagNrs[d], clustSubsetIndices[c]].shape)
          #   labelCurr = None
          #   if counterDiagLegend[longDiag[s]] == 0:
          #     labelCurr = plotTrajParams['diagLabels'][longDiag[s]]
          #     counterDiagLegend[longDiag[s]] += 1
          #
          #   pl.plot(longDPS[s], weightedDataMeanLong[s][:,c],
          #     c=plotTrajParams['diagColors'][longDiag[s]],
          #     label=labelCurr)

          if col == 0:
            ax.set_ylabel('$Z-score\ of\ biomarker$')

          if row == (nrRows - 1):
            print('disprogscore c=', c)
            ax.set_xlabel('$disease\ progression\ score$')
          else:
            pl.gca().set_xticklabels([])

          pl.xlim(np.min(dps), np.max(dps))

          # pl.ylim(-1.6,1)
          if 'ylimTrajWeightedDataMean' in plotTrajParams.keys() and not yLimUseData:
            pl.ylim(plotTrajParams['ylimTrajWeightedDataMean'])
          else:
            pl.ylim(yMinC[c] - delta[c], yMaxC[c] + delta[c])

          # if c == 3:
          #   fig.show()
          #   import pdb
          #   pdb.set_trace()

          # print('weightedDataMeanSC', weightedDataMeanSC[:,c])
          # print('fsCurrCT[c,:]', fsCurrCT[c,:])
          # print('xs', xs)
          # print('longDPS', [longDPS[s] for s in range(5)])
          # print('weightedDataMeanLong[s][:,c]', [weightedDataMeanLong[s][:,c] for s in range(5)])

    adjustCurrFig(plotTrajParams)
    pl.gcf().subplots_adjust(bottom=adjustBottomHeight)
    h, axisLabels = ax.get_legend_handles_labels()
    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )

    mng = pl.get_current_fig_manager()
    mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    pl.pause(0.05)
    return fig

  def plotTrajIndivBiomk(self, data, diag, labels, dps, longData, longDiag, longDPS, thetas, variances,
    plotTrajParams, trajFunc, nanMask, replaceFigMode=True, showConfInt=True,
    colorTitle=True, yLimUseData=False, adjustBottomHeight=0.25):

    figSizeInch = (plotTrajParams['SubfigClustMaxWinSize'][0]/100,
    plotTrajParams['SubfigClustMaxWinSize'][1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrRows = 3
    nrCols = 4

    nrSubj, nrBiomk = data.shape

    orderInd = range(nrBiomk)

    xs = np.linspace(np.min(dps), np.max(dps), 100)
    stdDevs = np.sqrt(variances)
    diagNrs = np.unique(diag)

    nrSubjToDisplay = nrSubj
    dataSubsetIndices = np.random.choice(np.array(range(nrSubj)), nrSubjToDisplay, replace = False)

    dataSubset = data[dataSubsetIndices, :]
    diagSubset = diag[dataSubsetIndices]
    dpsSubset = dps[dataSubsetIndices]

    nrSubjLong = len(longData)

    yMinB = np.min(data, axis=0)
    yMaxB = np.max(data, axis=0)
    delta = (yMaxB - yMinB) / 10

    fsCurrBT = np.zeros((nrBiomk, xs.shape[0]), np.float)

    dataNaNs = data.copy()
    dataNaNs[nanMask] = np.nan

    for row in range(nrRows):
      for col in range(nrCols):
        b = row * nrCols + col # clusterNr
        if b < nrBiomk:
          ax = pl.subplot(nrRows, nrCols, orderInd[b] + 1)
          ax.set_title('%s' % labels[orderInd[b]])
          if colorTitle:
            ax.title.set_color(plotTrajParams['clustCols'][b])

          fsCurrBT[b,:] = trajFunc(xs, thetas[b,:])

          pl.plot(xs, fsCurrBT[b,:], 'k-', linewidth=3.0) # label='sigmoid traj %d' % b
          if showConfInt:
            pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurrBT[b,:] - 1.9600 * stdDevs[b],
              (fsCurrBT[b,:] + 1.9600 * stdDevs[b])[::-1]]), alpha=.3, fc='b', ec='None')
                    #label='conf interval (1.96*std)')


          counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
          for s in range(nrSubjLong):
            #print(clustSubsetIndices, dataSubset[diagSubset == diagNrs[d], clustSubsetIndices[b]].shape)
            labelCurr = None
            if counterDiagLegend[longDiag[s]] == 0:
              labelCurr = plotTrajParams['diagLabels'][longDiag[s]]
              counterDiagLegend[longDiag[s]] += 1

            pl.plot(longDPS[s], longData[s][:,b],
              c=plotTrajParams['diagColors'][longDiag[s]],
              label=labelCurr)

          if col == 0:
            pl.ylabel('$Z-score\ of\ biomarker$')
          if row == nrRows - 1:
            pl.xlabel('$disease\ progression\ score$')

          pl.xlim(np.min(dps), np.max(dps))

          # pl.ylim(-1.6,1)
          if 'ylimTrajWeightedDataMean' in plotTrajParams.keys() and not yLimUseData:
            pl.ylim(plotTrajParams['ylimTrajWeightedDataMean'])
          else:
            pl.ylim(yMinB[b] - delta[b], yMaxB[b] + delta[b])



          print('biomk %d ' % b)
          print('Full CTL %f +/- %f', np.mean(data[diag == CTL,b]), np.std(data[diag == CTL,b]))
          print('Full AD %f +/- %f', np.mean(data[diag == AD, b]), np.std(data[diag == AD, b]))
          print('NaNs CTL %f +/- %f', np.nanmean(dataNaNs[diag == CTL,b]), np.nanstd(dataNaNs[diag == CTL,b]))
          print('NaNs AD %f +/- %f', np.nanmean(dataNaNs[diag == AD, b]), np.nanstd(dataNaNs[diag == AD, b]))
          print('data', data[:40,b])
          print('fsCurrBT[b,:]', fsCurrBT[b,:])
          print('xs', xs)
          print('longDPS', [longDPS[s] for s in range(3)])
          print('longData[s][:,b]', [longData[s][:,b] for s in range(3)])

          # if b == 5:
          #   print(ads)

    adjustCurrFig(plotTrajParams)
    pl.gcf().subplots_adjust(bottom=adjustBottomHeight)
    #fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    #print(h[2:4], labels[2:4])
    #legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    #legend = pl.legend(handles=h, loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=plotTrajParams['legendCols'], labelspacing=0. )
    # set the linewidth of each legend object
    # for i,legobj in enumerate(legend.legendHandles):
    #   legobj.set_linewidth(4.0)
    #   legobj.set_color(plotTrajParams['diagColors'][diagNrs[i]])

    mng = pl.get_current_fig_manager()
    # print(plotTrajParams['SubfigClustMaxWinSize'])
    # print(adsds)
    mng.resize(*plotTrajParams['SubfigClustMaxWinSize'])

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    #print("Plotting results .... ")
    pl.pause(0.05)
    return fig