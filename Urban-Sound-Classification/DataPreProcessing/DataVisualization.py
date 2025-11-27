import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import librosa
from .AudioManagement import (loadAudio)
from PIL import Image
import matplotlib.image as mpimg

def pastelizeColor(c:tuple, weight:float=None) -> np.ndarray:
    """
    # Descrição
        -> Clareia a cor de entrada misturando-a com branco, produzindo um efeito pastel.
    -----------------------------------------------------------------------------------
    := param: c - Cor original.
    := param: weight - Quantidade de branco para misturar (0 = cor total, 1 = branco total).
    """

    # Define um peso padrão
    weight = 0.5 if weight is None else weight

    # Inicializa um array com valores da cor branca para ajudar a criar a versão pastel da cor dada
    white = np.array([1, 1, 1])

    # Retorna uma tupla com os valores para a versão pastel da cor fornecida
    return mcolors.to_rgba((np.array(mcolors.to_rgb(c)) * (1 - weight) + white * weight))

def plotFeatureDistribution(df:pd.DataFrame=None, classFeature:str=None, forceCategorical:bool=None, pathsConfig:dict=None, featureDecoder:dict=None) -> None:
    """
    # Descrição
        -> Esta função plota a distribuição de uma feature (coluna) num dataset.
    -------------------------------------------------------------------------------
    := param: df - DataFrame do Pandas contendo os metadados do dataset.
    := param: feature - Feature do dataset para plotar.
    := param: forceCategorical - Força uma análise categórica numa feature numérica.
    := param: pathsConfig - Dicionário com caminhos importantes utilizados para armazenar alguns plots.
    := param: featureDecoder - Dicionário com a conversão entre o valor da coluna e o seu rótulo [De Inteiro para String].
    """

    # Verifica se um dataframe foi fornecido
    if df is None:
        print('O dataframe não foi fornecido.')
        return
    
    # Verifica se uma feature foi dada
    if classFeature is None:
        print('Falta uma feature para Analisar.')
        return

    # Verifica se a feature existe no dataset
    if classFeature not in df.columns:
        print(f"A feature '{classFeature}' não está presente no dataset.")
        return

    # Define o valor padrão
    forceCategorical = False if forceCategorical is None else forceCategorical

    # Define um caminho de ficheiro para armazenar o plot final
    if pathsConfig is not None:
        savePlotPath = pathsConfig['ExploratoryDataAnalysis'] + '/' + f'{classFeature}Distribution.png'
    else:
        savePlotPath = None

    # Define um tamanho de Figura
    figureSize = (8,5)

    # Verifica se o plot já foi calculado
    if savePlotPath is not None and os.path.exists(savePlotPath):
        # Carrega o ficheiro de imagem com o plot
        plot = mpimg.imread(savePlotPath)

        # Obtém as dimensões do plot em pixels
        height, width, _ = plot.shape

        # Define um valor de DPI utilizado para guardar o plot anteriormente
        dpi = 100

        # Cria uma figura com as mesmas dimensões exatas que o plot calculado anteriormente
        _ = plt.figure(figsize=(width / 2 / dpi, height / 2 / dpi), dpi=dpi)

        # Apresenta o plot
        plt.imshow(plot)
        plt.axis('off')
        plt.show()
    else:
        # Verifica o tipo da feature
        if pd.api.types.is_numeric_dtype(df[classFeature]):
            # Para features numéricas semelhantes a classes, podemos tratá-las como categorias
            if forceCategorical:
                # Cria uma figura
                _ = plt.figure(figsize=figureSize)

                # Obtém valores únicos e suas contagens
                valueCounts = df[classFeature].value_counts().sort_index()
                
                # Verifica se um Decodificador de feature foi dado e mapeia os valores se possível
                if featureDecoder is not None:
                    # Mapeia os valores inteiros para rótulos string
                    labels = valueCounts.index.map(lambda x: featureDecoder.get(x, x))
                    
                    # Inclina os rótulos do eixo x por 0 graus e ajusta o tamanho da fonte
                    plt.xticks(rotation=0, ha='center', fontsize=8)
                
                # Utiliza valores numéricos como rótulos de classe
                else:
                    labels = valueCounts.index

                # Cria um mapa de cores de verde para vermelho
                cmap = plt.get_cmap('RdYlGn_r')  # Mapa de cores invertido 'Vermelho-Amarelo-Verde' (verde para vermelho)
                colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plota as barras com cores de gradiente
                bars = plt.bar(labels.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plota a grelha atrás das barras
                plt.grid(True, zorder=1)

                # Adiciona texto (contagens de valores) a cada barra no centro com uma cor de fundo
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Utiliza uma cor mais clara como fundo para o texto
                    lighterColor = pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Adiciona título e rótulos
                plt.title(f'Distribuição de {classFeature}')
                plt.xlabel(f'Rótulos de {classFeature}', labelpad=20)
                plt.ylabel('Número de Amostras')
                
                # Guarda o plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Apresenta o plot
                plt.show()
            
            # Para features numéricas, utiliza um histograma
            else:
                # Cria uma figura
                plt.figure(figsize=figureSize)

                # Plota o histograma com cores de gradiente
                plt.hist(df[classFeature], bins=30, color='lightgreen', edgecolor='lightgrey', alpha=1.0, zorder=2)
                
                # Adiciona título e rótulos
                plt.title(f'Distribuição de {classFeature}')
                plt.xlabel(classFeature)
                plt.ylabel('Frequência')
                
                # Inclina os rótulos do eixo x por 0 graus e ajusta o tamanho da fonte
                plt.xticks(rotation=0, ha='center', fontsize=10)

                # Plota a grelha atrás das barras
                plt.grid(True, zorder=1)
                
                # Guarda o plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Apresenta o plot
                plt.show()

        # Para features categóricas, utiliza um gráfico de barras
        elif pd.api.types.is_categorical_dtype(df[classFeature]) or df[classFeature].dtype == object:
                # Cria uma figura
                plt.figure(figsize=figureSize)

                # Obtém valores únicos e suas contagens
                valueCounts = df[classFeature].value_counts().sort_index()
                
                # Cria um mapa de cores de verde para vermelho
                cmap = plt.get_cmap('viridis')  # Mapa de cores invertido 'Vermelho-Amarelo-Verde' (verde para vermelho)
                colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plota as barras com cores de gradiente
                bars = plt.bar(valueCounts.index.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plota a grelha atrás das barras
                plt.grid(True, zorder=1)

                # Adiciona texto (contagens de valores) a cada barra no centro com uma cor de fundo
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Utiliza uma cor mais clara como fundo para o texto
                    lighterColor = pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Adiciona título e rótulos
                plt.title(f'Distribuição de {classFeature}')
                plt.xlabel(f'Rótulos de {classFeature}', labelpad=20)
                plt.ylabel('Número de Amostras')
                
                # Inclina os rótulos do eixo x por 0 graus e ajusta o tamanho da fonte
                plt.xticks(rotation=25, ha='center', fontsize=8)

                # Guarda o plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Apresenta o plot
                plt.show()
        
        # Comportamento Desconhecido
        else:
            print(f"A feature '{classFeature}' não é suportada para plotagem.")

def plotFeatureDistributionByFold(df:pd.DataFrame=None, classFeature:str=None, foldFeature:str=None, pathsConfig:dict=None, featureDecoder:dict=None) -> None:
    """
    # Descrição
        -> Plota a distribuição de classes para cada fold no dataset.
    -----------------------------------------------------------------
    := param: df - DataFrame do Pandas contendo os metadados do dataset.
    := param: classFeature - A feature de classe do dataset.
    := param: foldFeature - A feature que indica o fold.
    := param: pathsConfig - Dicionário com caminhos importantes utilizados para armazenar alguns plots.
    := param: featureDecoder - Dicionário com a conversão entre o valor de classe e o seu rótulo.
    """

    # Verifica se DataFrame, feature de classe e feature de fold foram fornecidos
    if df is None or classFeature is None or foldFeature is None:
        print('DataFrame, feature de classe ou feature de fold está em falta.')
        return
    
    # Verifica se as features existem no DataFrame
    if classFeature not in df.columns or foldFeature not in df.columns:
        print(f"Ou '{classFeature}' ou '{foldFeature}' não está presente no dataset.")
        return
    
    # Define um caminho para guardar o plot
    if pathsConfig is not None:
        savePlotPath = pathsConfig['ExploratoryDataAnalysis'] + '/' + f'{classFeature}DistributionPerFold.png'
    else:
        savePlotPath = None

    # Define um tamanho de Figura
    figureSize = (18, 8)

    # Verifica se o plot já foi calculado
    if savePlotPath is not None and os.path.exists(savePlotPath):
        # Carrega o ficheiro de imagem com o plot
        plot = mpimg.imread(savePlotPath)

        # Obtém as dimensões do plot em pixels
        height, width, _ = plot.shape

        # Define um valor de DPI utilizado para guardar o plot anteriormente
        dpi = 300  

        # Cria uma figura com as mesmas dimensões exatas que o plot calculado anteriormente
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        # Apresenta o plot
        plt.imshow(plot)
        plt.axis('off')
        plt.show()

    else:
        # Obtém os folds únicos
        uniqueFolds = sorted(df[foldFeature].unique())
        
        # Configura uma grelha 2x5 para os subplots
        fig, axes = plt.subplots(2, 5, figsize=figureSize)
        fig.suptitle(f'Distribuição de {classFeature} em Cada Fold', fontsize=16, y=0.95)
        
        for i, fold in enumerate(uniqueFolds):
            # Filtra o DataFrame para o fold atual
            foldData = df[df[foldFeature] == fold]
            
            # Obtém contagens de valores para a feature de classe
            valueCounts = foldData[classFeature].value_counts().sort_index()
            
            # Utiliza o featureDecoder se fornecido
            if featureDecoder is not None:
                labels = valueCounts.index.map(lambda x: featureDecoder.get(x, x))
            else:
                labels = valueCounts.index
            
            # Cria um mapa de cores para as barras
            cmap = plt.get_cmap('viridis')
            colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]
            
            # Obtém o índice de linha e coluna para a grelha de subplots
            row, col = divmod(i, 5)
            
            # Plota as barras no subplot correto
            # bars = axes[row, col].bar(labels.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.4, zorder=2)

            # Ajusta as posições das barras e aumenta a sua espessura
            positions = np.arange(len(labels))  # Posições para as barras
            bars = axes[row, col].bar(positions, valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
            
            # Adiciona texto (contagens de valores) a cada barra no centro
            for j, bar in enumerate(bars):
                yval = bar.get_height()
                lighterColor = pastelizeColor(colors[j], weight=0.2)
                axes[row, col].text(bar.get_x() + bar.get_width() / 2,
                                    yval / 2,
                                    int(yval),
                                    ha='center',
                                    va='center',
                                    fontsize=7,
                                    color='black',
                                    bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Adiciona título e rótulos
            axes[row, col].set_title(f'[Fold] {fold}', fontsize=12)
            axes[row, col].set_xlabel(f'Rótulos de {classFeature}', fontsize=10)
            axes[row, col].set_ylabel('Número de Amostras', fontsize=10)
            axes[row, col].grid(True, zorder=1)
            axes[row, col].set_xticks(positions)
            axes[row, col].set_xticklabels(labels.astype(str), rotation=50, ha='right', fontsize=8)
            
        # Ajusta o layout para uma melhor apresentação
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Guarda o plot num ficheiro
        if savePlotPath is not None and not os.path.exists(savePlotPath):
            plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

        plt.show()

def plotAudioWave(df_audio:pd.DataFrame=None, audioSliceName:str=None, config:dict=None) -> None:
    """
    # Descrição
        -> Esta função plota a forma de onda do áudio selecionado. 
    ----------------------------------------------------------
    := param: df_audio - DataFrame do Pandas com os metadados do dataset.
    := param: audioSliceName - Identificação do Áudio dentro do dataset.
    := param: config - Dicionário com valores importantes utilizados durante tarefas de processamento de áudio dentro do projeto.
    := return: None, uma vez que estamos apenas a plotar a onda do áudio.
    """
    
    # Verifica se o dataframe com os metadados do dataset foi dado
    if df_audio is None:
        raise ValueError("Falta um DataFrame com os metadados do dataset!")

    # Verifica se um áudio foi selecionado
    if audioSliceName is None:
        raise ValueError("Falta um Áudio para plotar a forma de onda!")
    
    # Verifica se o dicionário de configuração foi fornecido
    if config is None:
        raise ValueError("Falta o dicionário de Configuração!")

    # Carrega o Áudio
    data = loadAudio(df_audio=df_audio, audioSliceName=audioSliceName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)

    # Configura o plot
    plt.figure(figsize=(12,4))

    # Plota a forma de onda
    librosa.display.waveshow(data, sr=config['SAMPLE_RATE'], color='dodgerblue', alpha=0.7)
    plt.fill_between(config['DURATION'], data, color='dodgerblue', alpha=0.3)  # Cor de preenchimento para melhor legibilidade

    # Adiciona título e rótulos
    plt.title('[Normalizado] Onda de Áudio', fontsize=16, pad=20)
    plt.xlabel("Tempo (s)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)

    # Personaliza a grelha e as espinhas
    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plotAudio1DimensionalFeature(audioData:np.ndarray=None, extracted1DimensionalFeature:np.ndarray=None, featureName:str=None, yLabel:str=None, color:str=None, config:dict=None) -> None:
    """
    # Descrição
        -> Esta função ajuda a plotar todas as Features Extraídas Unidimensionais.
    -------------------------------------------------------------------------
    := param: audioData - Série Temporal de Áudio dentro de um array numpy.
    := param: extracted1DimensionalFeature - Array Numpy com a Feature Unidimensional extraída dos AudioData.
    := param: featureName - Nome da Feature extraída do áudio.
    := param: yLabel - Rótulo para o eixo y.
    := param: color - Cor para a linha no plot da feature.
    := param: config - Dicionário com valores importantes utilizados durante tarefas de processamento de áudio dentro do projeto.
    := return: None, uma vez que estamos apenas a apresentar as features.
    """
   
    # Verifica se uma série temporal de áudio foi fornecida
    if audioData is None:
        raise ValueError("Falta uma Série Temporal de Áudio!")

    # Verifica se uma feature extraída foi fornecida
    if extracted1DimensionalFeature is None:
        raise ValueError("Falta Feature extraída")
    
    # Verifica forma da feature extraída
    if extracted1DimensionalFeature.shape[0] > 1:
        raise ValueError("Incompatibilidade de forma da feature extraída [Não corresponde a uma Feature Unidimensional]")

    # Verifica se um nome de feature foi dado
    if featureName is None:
        raise ValueError("Falta um Nome de Feature!")

    # Verifica se uma descrição de rótulo Y foi dado
    if yLabel is None:
        raise ValueError("Falta uma Descrição do Eixo Y!")

    # Verifica se uma configuração foi dada
    if config is None:
        raise ValueError("Falta um dicionário de configuração!")

    # Define uma cor padrão    
    color = 'r' if color is None else color

    # Aplana a feature extraída 1-D
    flattenedFeature = extracted1DimensionalFeature.flatten()
    
    # Aplica uma folha de estilos ao plot
    plt.style.use('seaborn-v0_8-pastel')

    # Cria uma figura
    plt.figure(figsize=(10, 6))
    
    # Plota a forma de onda do áudio
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audioData, sr=config['SAMPLE_RATE'], alpha=0.6, color='dodgerblue')
    plt.title('[Normalizado] Onda de Áudio', fontsize=14)
    plt.xlabel('Tempo (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True)
    
    # Plota a feature aplanada
    plt.subplot(2, 1, 2)
    frames = range(len(flattenedFeature))
    t = librosa.frames_to_time(frames, sr=config['SAMPLE_RATE'])
    plt.plot(t, flattenedFeature, color=color, label=featureName)
    plt.title(f'{featureName}', fontsize=14)
    plt.xlabel('Tempo (s)', fontsize=12)
    plt.ylabel(yLabel, fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    
    # Ajusta o layout
    plt.tight_layout(pad=3.0)
    plt.show()

def plotChromaFeatures(audioData:np.ndarray=None, config:dict=None) -> None:
    """
    # Descrição
        -> Esta função plota as features chroma do áudio.
    -------------------------------------------------------
    := param: audioData - Série Temporal de Áudio dentro de um array numpy.
    := param: config - Dicionário com valores importantes utilizados durante tarefas de processamento de áudio dentro do projeto.
    := return: None, uma vez que estamos apenas a plotar alguns dados.
    """
    
    # Verifica se uma série temporal de áudio foi fornecida
    if audioData is None:
        raise ValueError("Falta uma Série Temporal de Áudio!")

    # Verifica se uma configuração foi dada
    if config is None:
        raise ValueError("Falta um dicionário de configuração!")

    # Calcula as Features Chroma
    chroma_stft = librosa.feature.chroma_stft(y=audioData, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])
    
    # Cria uma figura
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', sr=config['SAMPLE_RATE'], cmap='PuBu')
    plt.colorbar()

    # Cria uma legenda para o eixo e adiciona um título
    plt.title('Features Chroma (STFT)', fontsize=14)
    plt.xlabel('Tempo (s)', fontsize=14)
    plt.ylabel('Chroma', fontsize=14)

    # Apresenta o plot
    plt.tight_layout()
    plt.show()

def plotMelSpectrogram(audioData:np.ndarray=None, sample_rate:int=None, n_mels:int=None, fmin:int=None, fmax=None) -> None:
    """
    # Descrição
        -> Plota um espectrograma Mel-frequency.
    -----------------------------------------
    Parâmetros:
    := param: audio - Série temporal de áudio.
    := param: sample_rate - Taxa de amostragem do áudio.
    := param: n_mels - Número de bandas Mel para gerar.
    := param: fmin - Frequência mínima para o banco de filtros Mel.
    := param: fmax - Frequência máxima para o banco de filtros Mel.
    := return: None, uma vez que estamos apenas a plotar o espectrograma mel.
    """

    # Verifica se uma série temporal de áudio foi fornecida
    if audioData is None:
        raise ValueError("Falta uma Série Temporal de Áudio!")
    
    # Verifica se uma Taxa de Amostragem foi dada
    if sample_rate is None:
        raise ValueError("Falta uma Taxa de Amostragem!")
    
    # Define Valores padrão
    n_mels = 128 if n_mels is None else n_mels
    fmin = 0 if fmin is None else fmin

    # Calcula o espectrograma Mel
    mel_spectrogram = librosa.feature.melspectrogram(y=audioData, sr=sample_rate, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Converte para dB para visualização
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Cria uma figura para o espectrograma
    plt.figure(figsize=(9, 5))
    ax = plt.axes()
    spec = librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax, ax=ax, cmap='PuBu')

    # Personaliza a barra de cores
    cbar = plt.colorbar(spec, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', rotation=270, labelpad=15, fontsize=12)

    # Adiciona ambos os rótulos e título
    plt.title('Espectrograma Mel-frequency', fontsize=16, pad=20)
    plt.xlabel("Tempo (s)", fontsize=14)
    plt.ylabel("Frequência (Mel)", fontsize=14)
    
    # Personaliza as marcas
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)

    plt.tight_layout(pad=2.0)
    plt.show()

def plotSpectralContrast(audioData:np.ndarray=None, config:dict=None) -> None:
    """
    # Descrição
        -> Esta função plota o contraste espectral do áudio.
    ---------------------------------------------------------
    := param: audioData - Série Temporal de Áudio dentro de um array numpy.
    := param: config - Dicionário com valores importantes utilizados durante tarefas de processamento de áudio dentro do projeto.
    := return: None, uma vez que estamos apenas a plotar alguns dados.
    """
    
    # Verifica se uma série temporal de áudio foi fornecida
    if audioData is None:
        raise ValueError("Falta uma Série Temporal de Áudio!")

    # Verifica se uma configuração foi dada
    if config is None:
        raise ValueError("Falta um dicionário de configuração!")

    # Calcula o Contraste Espectral
    spectralContrast = librosa.feature.spectral_contrast(y=audioData, sr=config['SAMPLE_RATE'])
    
    # Cria uma figura
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectralContrast, x_axis='time', sr=config['SAMPLE_RATE'], cmap='PuBu')
    plt.colorbar(label='Contraste Espectral (dB)')

    # Cria uma legenda para o eixo e adiciona um título
    plt.title('Contraste Espectral', fontsize=14)
    plt.xlabel('Tempo (s)', fontsize=14)
    plt.ylabel('Bandas de Frequência', fontsize=14)

    # Apresenta o plot
    plt.tight_layout()
    plt.show()
