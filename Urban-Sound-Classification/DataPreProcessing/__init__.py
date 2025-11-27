# Este pacote Python contém código utilizado durante as fases de Análise Exploratória de Dados e Pré-processamento de Dados.

# Definindo quais submódulos importar ao usar from <package> import *
__all__ = ["formatFilePath", "loadAudio",
           "plotFeatureDistribution"]

from .AudioManagement import (formatFilePath, loadAudio)
from .DataVisualization import (plotFeatureDistribution)

