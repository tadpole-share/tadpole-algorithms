# jupyter
https://github.com/tadpole-share/jupyter

## Setup
- use [miniconda3](https://docs.conda.io/en/latest/miniconda.html)
- make sure miniconda bin path is in $PATH
- conda env create -f environment.yml
- conda activate tadpole
- conda install -c conda-forge -n tadpole -y jupyterlab ipywidgets widgetsnbextension nodejs psutil
- jupyter labextension install @jupyter-widgets/jupyterlab-manager
- jupyter lab
