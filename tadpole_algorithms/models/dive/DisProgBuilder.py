from abc import ABC, abstractmethod

class DPMBuilder(ABC):

  def __init__(self):
    pass

  @abstractmethod
  def generate(self, dataIndices, expName, params):
    pass


class DPMInterface(ABC):

  @abstractmethod
  def run(self, runPart):
    raise NotImplementedError("Should have implemented this")

  @abstractmethod
  def stageSubjects(self, indices):
    raise NotImplementedError("Should have implemented this")

  @abstractmethod
  def stageSubjectsData(self, data):
    raise NotImplementedError("Should have implemented this")

  @abstractmethod
  def plotTrajectories(self, res):
    raise NotImplementedError("Should have implemented this")

  @abstractmethod
  def plotTrajSummary(self, res):
    raise NotImplementedError("Should have implemented this")


