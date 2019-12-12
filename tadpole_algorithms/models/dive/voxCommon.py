

def addParserArgs(parser):

  parser.add_argument('--fwhmLevel', dest="fwhmLevel", type=int,default=0,
                      help='full-width half max level: 0, 5, 10, 15, 20 or 25')

  parser.add_argument('--runIndex', dest='runIndex', type=int,
                      default=1,help='index of run instance/process')

  parser.add_argument('--nrProc', dest='nrProc', type=int,
                     default=1,help='# of processes')

  parser.add_argument('--modelToRun', dest='modelToRun', type=int,
                     help='index of model to run')

  parser.add_argument('--models', dest='models',default=13,
                     help='index of first experiment to run')

  parser.add_argument('--nrOuterIt', dest='nrOuterIt', type=int,
                     help='# of outer iterations to run, for estimating clustering probabilities')

  parser.add_argument('--nrInnerIt', dest='nrInnerIt', type=int,
                     help='# of inner iterations to run, for fitting the model parameters and subj. shifts')

  parser.add_argument('--nrClust', dest='nrClust', type=int,default=2,
                     help='# of clusters to fit')

  parser.add_argument('--cluster', action="store_true", default=False,
                     help='need to include this flag if runnin on cluster')

  parser.add_argument('--initClustering', dest="initClustering", default='hist',
                     help='initial clustering method: k-means or hist')

  parser.add_argument('--agg', dest='agg', type=int, default=0,
                     help='agg=1 => plot figures without using Xwindows, for use on cluster where the plots cannot be displayed '
                    ' agg=0 => plot with Xwindows (for use on personal machine)')

  parser.add_argument('--rangeFactor', dest='rangeFactor', type=float, default=2,
                     help='factor x such that min -= rangeDiff*x/10 and max += rangeDiff*x/10')

  parser.add_argument('--informPrior', dest='informPrior', type=int, default=1,
                     help='enables informative prior based on gamma and gaussian dist')

  parser.add_argument('--stdGammaAlpha', dest='stdGammaAlpha', type=float, default=0.0025,
                     help='std deviation of gamma prior on alpha')

  parser.add_argument('--stdBeta', dest='stdBeta', type=float, default=0.1,
                     help='std deviation of gaussian prior on beta')

  parser.add_argument('--lambdaMRF', dest='lambdaMRF', type=int, default=None,
                     help='lambda parameter for MRF, higher means more smoothing, lower means less. '
                          'only. If set to a fixed value, the optimisation for lambda is deactivated.')

  parser.add_argument('--reduceSpace', dest='reduceSpace', type=int, default=1,
                     help='choose not to save certain files in order to reduce space')

  parser.add_argument('--runPartStd', dest='runPartStd', default='RRRRR',
    help='choose which checkpoints to run: [initClust, modelFit, AIC/BIC, blender, theta_sampling] R=run L=load I=ignore')

  parser.add_argument('--runPartMain', dest='runPartMain', default='RII',
    help='choose which checkpoints to run: [mainPart, plot, stage]')

  parser.add_argument('--runPartCogCorr', dest='runPartCogCorr', default='R',
    help='choose which checkpoints to run: R or L')

  parser.add_argument('--runPartCogCorrMain', dest='runPartCogCorrMain', default='RRRRR',
    help='choose which checkpoints to run: [initClust, modelFit, AIC/BIC, blender, theta_sampling] R=run L=load I=ignore')



def initCommonVoxParams(args):

  from socket import gethostname
  import numpy as np
  import colorsys

  nrClust = args.get('nrClust')
  assert nrClust  > 1
  nrRows = int(np.sqrt(nrClust) * 0.95)
  nrCols = int(np.ceil(float(nrClust) / nrRows))
  assert (nrRows * nrCols >= nrClust)

  params = {}
  params['nrOuterIter'] = args.get('nrOuterIt')
  params['nrInnerIter'] = args.get('nrInnerIt')
  params['nrClust'] = nrClust
  params['runIndex'] = args.get('runIndex')
  params['nrProcesses'] = args.get('nrProc')
  params['rangeFactor'] = float(args.get('rangeFactor'))
  params['cluster'] = args.get('cluster')
  params['lambdaMRF'] = 1  # args.lambdaMRF # mrf lambda parameter, only used for MRF model

  # if args.lambdaMRF is not None:
  #   params['lambdaMRF'] = args.lambdaMRF
  #   params['fixMRF'] = True

  params['initClustering'] = args.get('initClustering')

  import aux
  priorNr = aux.setPrior(params, args.get('informPrior'), mean_gamma_alpha=1,
    std_gamma_alpha=args.get('stdGammaAlpha'), mu_beta=0, std_beta=args.get('stdBeta'))

  plotTrajParams = {}
  plotTrajParams['stagingHistNrBins'] = 20
  plotTrajParams['nrRows'] = nrRows
  plotTrajParams['nrCols'] = nrCols
  # plotTrajParams['freesurfPath'] = freesurfPath
  # plotTrajParams['blenderPath'] = blenderPath
  # plotTrajParams['homeDir'] = homeDir
  plotTrajParams['reduceSpace'] = args.get('reduceSpace')
  plotTrajParams['cluster'] = args.get('cluster')
  plotTrajParams['TrajSamplesFontSize'] = 12
  plotTrajParams['TrajSamplesAdjBottomHeight'] = 0.175
  plotTrajParams['trajSamplesPlotLegend'] = True

  if args.get('agg'):
    plotTrajParams['agg'] = True
  else:
    plotTrajParams['agg'] = False

  height = 700

  width = 1300
  if nrClust <= 4:
    heightClust = height / 2
  elif 4 < nrClust <= 6:
    heightClust = int(height * 2/3)
    width = 900
  else:
    heightClust = height

  plotTrajParams['SubfigClustMaxWinSize'] = (width, heightClust)
  plotTrajParams['SubfigVisMaxWinSize'] = (width, height)

  plotTrajParams['clustHuePoints'] = np.linspace(0,1,nrClust,endpoint=False)
  plotTrajParams['clustCols'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in plotTrajParams['clustHuePoints']]
  plotTrajParams['legendColsClust'] = min([nrClust, 4])

  return params, plotTrajParams