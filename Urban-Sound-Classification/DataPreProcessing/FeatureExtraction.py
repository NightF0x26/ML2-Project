import numpy as np
import pandas as pd
import os
import librosa
from .AudioManagement import (loadAudio)
        
def extractAllRawFeatures(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Descrição
        -> Esta função ajuda a extrair todas as características disponíveis
        das amostras de áudio do Fold selecionado no conjunto de dados
    --------------------------------------------------------------
    := param: audio_df - DataFrame do Pandas com os metadados do conjunto UrbamSound8K.
    := param: fold - Fold dos áudios para os quais queremos extrair características.
    := param: config - Dicionário com constantes usadas no processamento de áudio ao longo do projeto.
    := param: pathsConfig - Dicionário com caminhos de arquivos para organizar os resultados do projeto.
    := return: None, pois estamos apenas extraindo dados.
    """

    # Verifica se o dataframe já foi computado
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['All-Raw-Features']):
        # Inicializa uma lista para armazenar o conteúdo extraído
        data = []

        # Obtém os nomes dos arquivos de áudio do fold selecionado
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Itera por todos os áudios dentro do fold selecionado
        for audioFileName in foldAudios:
            # Carrega o áudio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # [Extrai Características]

            # [Características 1-Dimensionais]
            # Taxa de Cruzamento por Zero - Mede a taxa de mudança de sinal no áudio
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y=audio)[0]

            # Centroide Espectral - Centro de massa do espectro
            spectralCentroid = librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Largura de Banda Espectral - Largura da banda de frequências principais
            spectralBandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Planicidade Espectral - Medida de quanto o espectro é plano ou pontiagudo
            spectralFlatness = librosa.feature.spectral_flatness(y=audio)[0]

            # Roll-off Espectral - Frequência abaixo da qual está concentrada a energia do espectro
            spectralRolloff = librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Energia RMS - Raiz da Média Quadrática (Root Mean Square) da amplitude
            rmsEnergy = librosa.feature.rms(y=audio)[0]

            # [Características 2-Dimensionais]
            # MFCCs - Coeficientes de Frequência Mel Cepstral (baseados na percepção auditiva humana)
            mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])
            
            # Chroma STFT - Representação de energia em 12 classes de pitch
            chromaSTFT = librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])

            # Espectrograma Mel - Representação tempo-frequência da energia no espaço Mel
            melSpectrogram = librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE'])

            # Contraste Espectral - Medida de diferença entre os picos e vales do espectro
            spectralContrast = librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])

            # Computa e adiciona as características extraídas à lista de dados
            data.append({
                # Detalhes do Áudio
                'audio':audioFileName,
                'fold':fold,

                # Características 1-Dimensionais
                'Zero-Crossing Rate':zeroCrossingRate,
                'Spectral Centroid':spectralCentroid,
                'Spectral Bandwidth':spectralBandwidth,
                'Spectral Flatness':spectralFlatness,
                'Spectral Roll-off':spectralRolloff,
                'RMS Energy':rmsEnergy,
                
                # Características 2-Dimensionais
                'MFCC':mfcc,
                'Chroma STFT':chromaSTFT,
                'Mel Spectrogram':melSpectrogram,
                'Spectral Contrast':spectralContrast,

                # Alvo - Classe de som urbano
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Cria um DataFrame com os dados coletados
        df = pd.DataFrame(data)

        # Salva o DataFrame em formato pickle para carregamento rápido posterior
        df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}']['All-Raw-Features'])

def extractRawFeatures1D(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Descrição
        -> Esta função ajuda a extrair todas as características 1D brutas
        das amostras de áudio do Fold selecionado no conjunto de dados,
        além de normalizá-las e deixá-las prontas para uso.
    ------------------------------------------------------------------
    := param: audio_df - DataFrame do Pandas com os metadados do conjunto UrbamSound8K.
    := param: fold - Fold dos áudios para os quais queremos extrair características.
    := param: config - Dicionário com constantes usadas no processamento de áudio ao longo do projeto.
    := param: pathsConfig - Dicionário com caminhos de arquivos para organizar os resultados do projeto.
    := return: None, pois estamos apenas extraindo dados.
    """

    # Verifica se o dataframe já foi computado
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['1D-Raw-Features']):
        # Inicializa uma lista para armazenar o conteúdo extraído
        data = []

        # Obtém os nomes dos arquivos de áudio do fold selecionado
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Itera por todos os áudios dentro do fold selecionado
        for audioFileName in foldAudios:
            # Carrega o áudio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # Extrai características
            # [Características 1D]

            # Taxa de Cruzamento por Zero - Mede a taxa de mudança de sinal no áudio
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y=audio)[0]

            # Centroide Espectral - Centro de massa do espectro
            spectralCentroid = librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Largura de Banda Espectral - Largura da banda de frequências principais
            spectralBandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Planicidade Espectral - Medida de quanto o espectro é plano ou pontiagudo
            spectralFlatness = librosa.feature.spectral_flatness(y=audio)[0]

            # Roll-off Espectral - Frequência abaixo da qual está concentrada a energia do espectro
            spectralRolloff = librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Energia RMS - Raiz da Média Quadrática (Root Mean Square) da amplitude
            rmsEnergy = librosa.feature.rms(y=audio)[0]

            # Adiciona as características extraídas à lista de dados
            data.append({
                # Detalhes do Áudio
                'audio':audioFileName,
                'fold':fold,

                # Características 1D
                'Zero-Crossing Rate': zeroCrossingRate,
                'Spectral Centroid': spectralCentroid,
                'Spectral Bandwidth':spectralBandwidth,
                'Spectral Flatness':spectralFlatness,
                'Spectral Roll-off':spectralRolloff,
                'RMS Energy':rmsEnergy,

                # Alvo - Classe de som urbano
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Cria um DataFrame com os dados coletados
        df = pd.DataFrame(data)

        # Salva o DataFrame em formato pickle para carregamento rápido posterior
        df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}']['1D-Raw-Features'])

def extractRawFeatures2D(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Descrição
        -> Esta função ajuda a extrair todas as características 2D brutas
        das amostras de áudio do Fold selecionado no conjunto de dados,
        além de normalizá-las e deixá-las prontas para uso.
    ------------------------------------------------------------------
    := param: audio_df - DataFrame do Pandas com os metadados do conjunto UrbamSound8K.
    := param: fold - Fold dos áudios para os quais queremos extrair características.
    := param: config - Dicionário com constantes usadas no processamento de áudio ao longo do projeto.
    := param: pathsConfig - Dicionário com caminhos de arquivos para organizar os resultados do projeto.
    := return: None, pois estamos apenas extraindo dados.
    """

    # Verifica se o dataframe já foi computado
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['2D-Raw-Features']):
        # Inicializa uma lista para armazenar o conteúdo extraído
        data = []

        # Obtém os nomes dos arquivos de áudio do fold selecionado
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Itera por todos os áudios dentro do fold selecionado
        for audioFileName in foldAudios:
            # Carrega o áudio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # [Características 2-Dimensionais]

            # MFCCs - Coeficientes de Frequência Mel Cepstral (baseados na percepção auditiva humana)
            mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])
            
            # Chroma STFT - Representação de energia em 12 classes de pitch
            chromaSTFT = librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])

            # Espectrograma Mel - Representação tempo-frequência da energia no espaço Mel
            melSpectrogram = librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE'])

            # Contraste Espectral - Medida de diferença entre os picos e vales do espectro
            spectralContrast = librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])

            # Computa e adiciona as características extraídas à lista de dados
            data.append({
                # Detalhes do Áudio
                'audio':audioFileName,
                'fold':fold,
                
                # Características 2-Dimensionais
                'MFCC':mfcc,
                'Chroma STFT':chromaSTFT,
                'Mel Spectrogram':melSpectrogram,
                'Spectral Contrast':spectralContrast,

                # Alvo - Classe de som urbano
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Cria um DataFrame com os dados coletados
        df = pd.DataFrame(data)

        # Salva o DataFrame em formato pickle para carregamento rápido posterior
        df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}']['2D-Raw-Features'])
        
def getFeaturesDetails(df:pd.DataFrame, intervalStep:int) -> list[dict]:
        """
        # Descrição
            -> Com base nas características extraídas da primeira amostra de áudio, calcula a quantidade de
            componentes que precisamos para particionar cada característica considerando o possível resíduo.
        -----------------------------------------------------------------------------------------------
        := param: df - DataFrame do Pandas do qual queremos extrair os detalhes das características.
        := param: stepInterval - Tamanho da janela de segmentação que queremos considerar ao criar componentes para as características extraídas.
        := return: Uma lista com todos os dados sobre o formato dos dados na próxima etapa.
        """

        # Seleciona as colunas importantes para extrair os detalhes
        cols = df.columns[2:len(df.columns) - 2]

        # Lista para armazenar os detalhes das colunas a processar
        columnsDetails = []

        # Itera sobre as características do DataFrame
        for feature in cols:
            # Analisa o formato do array da característica atual
            length = len(df.iloc[0][feature])

            # Calcula o número de componentes para a característica atual
            numComponents = length // intervalStep
            residueSize = length / intervalStep

            # Atualiza a lista inicial com os dados calculados
            columnsDetails.append({
                'feature':feature.replace('-', '_').replace(' ', '_'),
                'totalComponents':numComponents,
                'residueSize':residueSize
            })
        
        # Retorna a lista com os detalhes de todas as características
        return columnsDetails

def processRawFeatures(fold:int, intervalStep:int, featuresDimensionality:str, pathsConfig:dict) -> None:
    """
    # Descrição
        -> Este método permite processar as características brutas extraídas anteriormente em
        múltiplos componentes com base em pequenas partições dos dados e algumas métricas.
    -------------------------------------------------------------------------------------
    := param: fold - Fold no qual queremos processar as características extraídas.
    := param: intervalStep - Tamanho da janela de segmentação que queremos considerar ao criar componentes para as características extraídas.
    := param: featureDimensionality - Dimensionalidade dos dados a processar ("1D" ou "2D").
    := return: None, pois estamos apenas atualizando um atributo da classe.
    """

    # Verifica a integridade da dimensionalidade escolhida
    if featuresDimensionality not in ["1D", "2D"]:
        raise ValueError("Dimensionalidade de característica inválida escolhida")

    # Cria uma variável para o dataframe processado
    processed_df = None

    # Se o DataFrame com os dados processados ainda não foi computado
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}'][f'{featuresDimensionality}-Processed-Features']):

        # Carrega o conjunto de dados com as características brutas e seleciona as colunas importantes
        df = pd.read_pickle(pathsConfig['Datasets'][f'Fold-{fold}'][f'{featuresDimensionality}-Raw-Features'])
        featuresToProcess = df.columns[2:len(df.columns) - 2]

        # Busca os detalhes das colunas
        columnDetails = getFeaturesDetails(df, intervalStep)

        # Itera linha por linha e processa cada vetor extraído para obter estatísticas
        # Resultado: múltiplas colunas [MEAN_F1, MEDIAN_F1, STD_F1, MEAN_F2, MEDIAN_F2, STD_F2, ...]
        for index, row in df.iterrows():
            # Cria um novo dicionário para uma nova linha no DataFrame
            audioSampleData = {'audio':row['audio'], 'fold':fold}

            # Cria um índice de característica para acompanhar a característica atual analisada
            featureIdx = 0

            # Processamento baseado na dimensionalidade das características
            if featuresDimensionality == "1D":
                # Itera pelas características 1D
                for feature in featuresToProcess:
                    # Busca o array na célula atual
                    featureArray = row[feature]

                    # Cria os componentes para os dados 1D
                    for currentComponent in range(1, columnDetails[featureIdx]['totalComponents'] + 1):
                        if currentComponent == columnDetails[featureIdx]['totalComponents'] - 1: 
                            audioSampleData.update({
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Mean":np.mean(featureArray[currentComponent*intervalStep :]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Median":np.median(featureArray[currentComponent*intervalStep :]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Std":np.std(featureArray[currentComponent*intervalStep :])
                            })
                        else:
                            audioSampleData.update({
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Mean":np.mean(featureArray[(currentComponent - 1)*intervalStep : currentComponent*intervalStep]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Median":np.median(featureArray[(currentComponent - 1)*intervalStep : currentComponent*intervalStep]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Std":np.std(featureArray[(currentComponent - 1)*intervalStep : currentComponent*intervalStep])
                            })
                    
                    # Incrementa o índice da característica sendo processada
                    featureIdx += 1
            
            elif featuresDimensionality == "2D":
                # Itera pelas características 2D
                for feature in featuresToProcess:
                    # Busca e converte o array na célula atual
                    featureArray = np.mean(row[feature], axis=1)

                    # Atualiza os dados da amostra de áudio com todos os componentes calculados previamente durante a extração de características
                    for componentIdx, component in enumerate(featureArray):
                        audioSampleData.update({
                            f"{columnDetails[featureIdx]['feature']}_{componentIdx}":component
                        })
                    
                    # Incrementa o índice da característica sendo processada
                    featureIdx += 1

            # Adiciona o rótulo alvo
            audioSampleData.update({
                'target':row['target']
            })

            # Verifica se já temos um DataFrame
            if processed_df is None:
                # Cria um novo do zero
                processed_df = pd.DataFrame([audioSampleData])
            else:
                # Cria um novo DataFrame com a nova entrada de áudio processada
                newLine = pd.DataFrame([audioSampleData])

                # Concatena o novo DataFrame com o anterior
                processed_df = pd.concat([processed_df, newLine], ignore_index=True)
        
        # Salva os dados processados em formato pickle para carregamento rápido posterior
        processed_df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}'][f'{featuresDimensionality}-Processed-Features'])

def extractMFCCs(audio_df:pd.DataFrame, raw:bool, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Descrição
        -> Esta função ajuda a extrair os MFCCs das amostras de áudio
        do Fold selecionado no conjunto de dados.
    --------------------------------------------------------------------
    := param: audio_df - DataFrame do Pandas com os metadados do conjunto UrbamSound8K.
    := param: raw - Valor booleano que determina se vamos trabalhar com dados brutos ou não.
    := param: fold - Fold dos áudios para os quais queremos extrair características.
    := param: config - Dicionário com constantes usadas no processamento de áudio ao longo do projeto.
    := param: pathsConfig - Dicionário com caminhos de arquivos para organizar os resultados do projeto.
    := return: None, pois estamos apenas extraindo dados.
    """
    
    # Define um valor padrão para o booleano raw
    raw = False if raw is None else raw

    # Define o caminho do arquivo baseado no tipo de dados (raw ou processado)
    if raw:
        mfccsFilePath = pathsConfig['Datasets'][f'Fold-{fold}']['2D-Raw-MFCCs']
    else:
        mfccsFilePath = pathsConfig['Datasets'][f'Fold-{fold}']['1D-Processed-MFCCs']

    # Verifica se o dataframe já foi computado
    if not os.path.exists(mfccsFilePath):
        # Inicializa uma lista para armazenar o conteúdo extraído
        data = []

        # Obtém os nomes dos arquivos de áudio do fold selecionado
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Itera por todos os áudios dentro do fold selecionado
        for audioFileName in foldAudios:
            # Carrega o áudio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # Define um dicionário para os dados do áudio atual
            audioData = {
                # Detalhes do Áudio
                'audio':audioFileName,
                'fold':fold,
            }

            # Computa e processa os MFCCs baseado no tipo de dados
            if raw: 
                # Dados brutos - Mantém a forma completa 2D
                mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])

                # Atualiza os dados do áudio
                audioData.update({
                    # MFCC
                    'MFCC':mfcc,
                })

            else: 
                # Dados processados - Calcula a média de cada coeficiente ao longo do tempo
                mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC']), axis=1)

                # Atualiza os dados da amostra de áudio com todos os componentes dos MFCCs
                for componentIdx, component in enumerate(mfcc):
                    audioData.update({
                        f"MFCC_{componentIdx}":component
                    })

            # Adiciona o rótulo alvo (classe de som urbano)
            audioData.update({
                # Alvo
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

            # Adiciona os dados do áudio à lista
            data.append(audioData)

        # Cria um DataFrame com os dados coletados
        df = pd.DataFrame(data)

        # Salva o DataFrame em formato pickle para carregamento rápido posterior
        df.to_pickle(mfccsFilePath)

