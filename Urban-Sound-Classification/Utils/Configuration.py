def loadConfig() -> dict:
    """
    # Descrição
        -> Esta função visa armazenar todos os parâmetros relacionados à configuração utilizados dentro do projeto.
    ----------------------------------------------------------------------------------------------------------------
    := retorno: Dicionário com algumas das constantes/valores importantes utilizados no projeto.
    """

    # Valores Computados
    sampleRate = 44100  # Taxa mais elevada para capturar áudios de alta resolução como os que vêm de ondas harmónicas.
    hopLength = round(sampleRate * 0.0125)  # Número de amostras para avançar entre quadros
    windowLength = round(sampleRate * 0.023)  # Número de amostras utilizadas em cada quadro para análise de frequência
    timeSize = 4 * sampleRate // hopLength + 1  # Número de quadros temporais após aplicar hop length e janelamento
    return {
        "DURATION": 4,  # Comprimento de cada amostra de áudio no conjunto de dados.
        "SAMPLE_RATE": sampleRate,  # Número de amostras de áudio por segundo quando convertendo de sinal contínuo para digital
        "HOP_LENGTH": hopLength,  # Número de amostras para avançar entre quadros
        "WINDOW_LENGTH": windowLength,  # Número de amostras utilizadas em cada quadro para análise de frequência, ou comprimento da janela na qual a Transformada de Fourier é aplicada.
        "N_FFT": 2**10,  # Comprimento do sinal com janela após preenchimento com zeros
        "TIME_SIZE": timeSize,  # Número de quadros temporais ou segmentos nos quais o áudio será dividido após aplicar hop length e janelamento.
        "N_CHROMA": 12,  # Número de classes de altura (ex: C, C#, D, etc.) na representação de característica chroma.
        "N_MFCC": 13,  # Número de Coeficientes Cepstrais de Frequência Mel (MFCCs) a serem extraídos
    }


def createDatasetsFolderPaths(numberFolds: int) -> dict:
    """
    # Descrição
        -> Esta função ajuda a configurar todos os caminhos nos quais vamos armazenar as características extraídas.
    ----------------------------------------------------------------------------------------------------------------
    := param: numberFolds -  Número de folds que serão considerados.
    := retorno: Dicionário com todos os caminhos dos datasets configurados.
    """

    # Cria um dicionário principal para armazenar todos os caminhos
    datasetsPaths = {}

    for foldIdx in range(1, numberFolds + 1):
        # Define os caminhos para o fold atual
        currentFoldDatasetsPaths = {
            # Todas as Características Disponíveis para processar as Amostras de Áudio
            "All-Raw-Features": f"./Datasets/Fold-{foldIdx}/All-Raw-Features.pkl",
            # Ficheiros para armazenar Características 1-Dimensionais
            "1D-Raw-Features": f"./Datasets/Fold-{foldIdx}/1D-Raw-Features.pkl",
            "1D-Processed-Features": f"./Datasets/Fold-{foldIdx}/1D-Processed-Features.pkl",
            # Ficheiros para armazenar Características 2-Dimensionais
            "2D-Raw-Features": f"./Datasets/Fold-{foldIdx}/2D-Raw-Features.pkl",
            "2D-Processed-Features": f"./Datasets/Fold-{foldIdx}/2D-Processed-Features.pkl",
            # Ficheiros para armazenar os MFCCs
            "2D-Raw-MFCCs": f"./Datasets/Fold-{foldIdx}/2D-Raw-MFCCs.pkl",
            "1D-Processed-MFCCs": f"./Datasets/Fold-{foldIdx}/1D-Processed-MFCCs.pkl",
        }

        # Atualiza o dicionário principal
        datasetsPaths.update({f"Fold-{foldIdx}": currentFoldDatasetsPaths})

    # Adiciona o caminho para aprendizado por transferência
    datasetsPaths.update({"transfer": "./Datasets/transfer.pkl"})

    # Retorna o dicionário com todos os caminhos para os datasets
    return datasetsPaths


def createModelResultsFolderPaths(
    modelName: str, numberTests: int, numberFolds: int
) -> dict:
    """ "
    # Descrição
        -> Esta função ajuda a criar todos os caminhos para armazenar os resultados experimentais de um modelo dado.
    ----------------------------------------------------------------------------------------------------------------
    := param: modelName - Nome do modelo.
    := param: numberTests - Número de testes que serão realizados no modelo.
    := param: numberFolds -  Número de folds que serão considerados.
    := retorno: Dicionário com todos os caminhos apropriados formatados para os resultados experimentais do modelo.
    """

    # Cria um dicionário principal para armazenar todos os caminhos para os resultados de um modelo
    testsData = {}

    # Itera pela quantidade de testes a realizar
    for testIdx in range(1, numberTests + 1):
        # Cria um dicionário para os caminhos dos resultados de cada fold para o teste atual
        foldsData = {}

        # Itera por cada fold
        for foldIdx in range(1, numberFolds + 1):
            # Cria os caminhos
            foldsData.update(
                {
                    f"Fold-{foldIdx}": {
                        "History": f"./ExperimentalResults/ModelPerformanceResults/{modelName}/Test-{testIdx}/Fold-{foldIdx}/history.pkl",
                        "Model": f"./ExperimentalResults/ModelPerformanceResults/{modelName}/Test-{testIdx}/Fold-{foldIdx}/model.keras",
                    }
                }
            )
        # Atualiza o dicionário principal
        testsData.update({f"Test-{testIdx}": foldsData})

    # Retorna os Caminhos para os resultados experimentais do modelo
    return testsData


def loadPathsConfig() -> dict:
    """
    # Descrição
        -> Esta função visa armazenar todos os parâmetros relacionados à configuração de caminhos utilizados dentro do projeto.
    -----------------------------------------------------------------------------------------------------------------------
    := retorno: Dicionário com alguns dos caminhos de ficheiros importantes do projeto.
    """
    return {
        "ExploratoryDataAnalysis": "./ExperimentalResults/ExploratoryDataAnalysis",
        "Datasets": createDatasetsFolderPaths(numberFolds=10),
        "ModelDevelopmentAndEvaluation": {
            "MLP": createModelResultsFolderPaths(
                modelName="MLP", numberTests=4, numberFolds=10
            ),
            "CNN": createModelResultsFolderPaths(
                modelName="CNN", numberTests=5, numberFolds=10
            ),
            "YAMNET": createModelResultsFolderPaths(
                modelName="YAMNET", numberTests=4, numberFolds=10
            ),
            "ResNet": createModelResultsFolderPaths(
                modelName="ResNet", numberTests=3, numberFolds=10
            ),
            "yamnet-train": "./ExperimentalResults/ModelDevelopmentAndEvaluation/yamnet/train.pkl",
            "yamnet-test": "./ExperimentalResults/ModelDevelopmentAndEvaluation/yamnet/test.pkl",
        },
    }
