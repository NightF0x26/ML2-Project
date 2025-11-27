# Este Pacote Python contém código utilitário usado em várias tarefas diversas ao longo do projeto

# Definindo quais submódulos importar ao usar from <package> import *
__all__ = ["loadConfig", "loadPathsConfig"]

from .Configuration import (loadConfig, loadPathsConfig)