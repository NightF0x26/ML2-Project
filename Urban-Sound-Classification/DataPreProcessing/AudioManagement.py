import librosa as libr
import numpy as np
import pandas as pd
from IPython import display


def formatFilePath(audioFold:int, audioName:str) -> str:
    """
    # Descrição
        -> Cria o caminho do arquivo para acessar corretamente um arquivo de áudio do dataset UrbanSound8K.
    -----------------------------------------------------------------------------------------
    := param: audioFold - pasta ao qual a amostra de áudio pertence dentro do dataset.
    := param: audioName - Nome do arquivo de áudio dentro do dataset.
    := return: String que aponta para o arquivo correto.
    """
    # Retorna o caminho do arquivo de áudio no UrbanSound8K
    return f'./UrbanSound8K/audio/fold{audioFold}/{audioName}'


def loadAudio(df_audio:pd.DataFrame, audioSliceName:int, audioDuration:int, targetSampleRate:int, usePadding:bool) -> np.ndarray:
    """
    # Descrição
        -> Carrega um arquivo de áudio do dataset.
    -------------------------------------------
    := param: df_audio - DataFrame do Pandas com os metadados do dataset.
    := param: audioSliceName - Identificação do áudio dentro do dataset.
    := param: audioDuration - Duração a ser considerada do áudio.
    := param: targetSampleRate - Taxa de amostragem alvo para o áudio.
    := param: usePadding - Se deve ou não realizar zero padding no áudio reamostrado na taxa alvo.
    := return: Objeto de áudio.
    """

    # Seleciona a entrada do áudio pelo nome do arquivo
    df_audio_selectedAudio = df_audio[df_audio['slice_file_name'] == audioSliceName]

    # Obtém o índice da linha da entrada
    idx = df_audio_selectedAudio.index.values.astype(int)[0]

    # Obtém o fold do áudio
    audioFold = df_audio_selectedAudio['fold'][idx]

    # Formata o caminho do arquivo
    audioFilePath = formatFilePath(audioFold, audioSliceName)

    # Carrega o áudio usando a taxa de amostragem original
    audioTimeSeries, samplingRate = libr.load(audioFilePath, duration=audioDuration, sr=None)

    # Reamostra o áudio para a taxa de amostragem alvo
    audioTimeSeries = libr.resample(audioTimeSeries, orig_sr=samplingRate, target_sr=targetSampleRate)

    # Realiza padding no áudio para que todas as séries temporais tenham o mesmo comprimento
    if usePadding:
        audioTimeSeries = libr.util.fix_length(data=audioTimeSeries, size=audioDuration*targetSampleRate, mode='constant')

    # Retorna o áudio (com padding, se aplicável)
    return audioTimeSeries


def showcaseAudio(df_audio:pd.DataFrame, audioSliceName:int) -> display.Audio:
    """
    # Descrição
        -> Cria um player de áudio simples para o arquivo selecionado do dataset.
    ----------------------------------------------------------------------------
    := param: df_audio - DataFrame do Pandas com os metadados do dataset.
    := param: audioSliceName - Identificação do áudio dentro do dataset.
    := return: Objeto de áudio que permite ouvir o arquivo selecionado.
    """

    # Seleciona a entrada do áudio pelo nome do arquivo
    df_audio_selectedAudio = df_audio[df_audio['slice_file_name'] == audioSliceName]

    # Obtém o índice da linha da entrada
    idx = df_audio_selectedAudio.index.values.astype(int)[0]

    # Obtém o fold do áudio
    audioFold = df_audio_selectedAudio['fold'][idx]

    # Formata o caminho do arquivo
    audioFilePath = formatFilePath(audioFold, audioSliceName)

    # Retorna o player de áudio para escutar o arquivo
    return display.Audio(audioFilePath)

