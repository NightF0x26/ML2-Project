# =====================================================================
# Módulo de Visualização de Desempenho de Modelos de Aprendizado de Máquina
# =====================================================================
# Este módulo fornece funções para visualizar e analisar o desempenho de
# modelos de classificação de audio urbano, incluindo gráficos de desempenho,
# matrizes de confusão, e análises de redução de dimensionalidade.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.callbacks.history import History
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, RobustScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scikit_posthocs as sp




# =====================================================================
# Função: plotNetworkTrainingPerformance
# =====================================================================
def plotNetworkTrainingPerformance(
    confusionMatrix: np.ndarray, title: str, trainHistory: History, targetLabels=None
) -> None:
    """
    # Descrição
        -> Esta função ajuda a visualizar o desempenho da rede
        durante o treino através da variação da perda e da Exatidão.
        Plota 3 visualizações: Exatidão, perda e matriz de confusão.
    
    # Parâmetros
        -> confusionMatrix: Matriz de confusão obtida do modelo fornecido
        -> title: Título principal para o conjunto de gráficos
        -> trainHistory: Dados do histórico de treino (períodos, Exatidão, perda)
        -> targetLabels: Rótulos alvo do conjunto UrbanSound8k
    
    # Retorno
        -> None, pois estamos apenas plotando dados
    """

    # Cria uma figura com 3 subgráficos lado a lado (1 linha, 3 colunas)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Define o título geral para toda a figura
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # ===== SUBPLOT 1: Exatidão do Modelo =====
    # Plota a Exatidão de treino e validação ao longo dos períodos
    ax1.plot(trainHistory["accuracy"], label="Exatidão Treino")
    ax1.plot(trainHistory["val_accuracy"], label="Exatidão Validação")
    ax1.set_title("Exatidão do Modelo")
    ax1.set_ylabel("Exatidão")
    ax1.set_xlabel("Período")
    ax1.legend(loc="lower right")
    # Nota: A diferença entre treino e validação indica possível overfitting

    # ===== SUBPLOT 2: perda do Modelo =====
    # Plota a perda (erro) de treino e validação ao longo dos períodos
    ax2.plot(trainHistory["loss"], label="Perda Treino")
    ax2.plot(trainHistory["val_loss"], label="Perda Validação")
    ax2.set_title("Perda do Modelo")
    ax2.set_ylabel("Perda")
    ax2.set_xlabel("Período")
    ax2.legend(loc="upper right")
    # Nota: Convergência simultânea indica bom aprendizado

    # ===== SUBPLOT 3: Matriz de Confusão =====
    # Visualiza as previsões corretas e erradas para cada classe
    plotConfusionMatrix(confusionMatrix, targetLabels=targetLabels, ax=ax3)

    # Ajusta espaçamento entre subgráficos
    plt.tight_layout()
    plt.show()



# =====================================================================
# Função: plotConfusionMatrix
# =====================================================================
def plotConfusionMatrix(
    confusionMatrix, title="Matriz de Confusão", targetLabels=None, ax=None
):
    """
    # Descrição
        -> Visualiza a matriz de confusão de um modelo de classificação.
        Mostra quantas amostras foram classificadas corretamente ou incorretamente.
    
    # Parâmetros
        -> confusionMatrix: Array 2D com as contagens de classificações
        -> title: Título para o gráfico da matriz
        -> targetLabels: Nomes das classes para os eixos
        -> ax: Eixo matplotlib para desenhar (se None, cria nova figura)
    """
    # Se nenhum eixo foi fornecido, cria um novo
    if ax is None:
        fig, ax = plt.subplots()

    # Plota a matriz de confusão como um mapa de calor (heatmap)
    # - annot=True: mostra os valores numéricos nas células
    # - fmt="d": formata como números inteiros
    # - cmap="Blues": escala de cores de azul (mais escuro = valores maiores)
    sns.heatmap(
        confusionMatrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=targetLabels,
        yticklabels=targetLabels,
    )

    # Configuração dos eixos e título
    ax.set_title(title)
    ax.set_xlabel("Rótulos Previstos")  # Classes que o modelo previu
    ax.set_ylabel("Rótulos Verdadeiros")  # Classes reais (corretas)




# =====================================================================
# Função: plotCritialDifferenceDiagram
# =====================================================================
def plotCritialDifferenceDiagram(
    matrix: np.ndarray = None, colors: dict = None
) -> None:
    """
    # Descrição
        -> Plota o Diagrama de Diferença Crítica (Critical Difference Diagram).
        -> Este diagrama compara múltiplos modelos usando teste post-hoc de Nemenyi
           para determinar se há diferenças significativas entre eles.
    
    # Parâmetros
        -> matrix: DataFrame com as Exatidãos obtidas pelos modelos
        -> colors: Dicionário que associa cada coluna do df a uma cor para usar no diagrama
    
    # Retorno
        -> None, pois estamos apenas plotando um diagrama
    
    # Estatística
        -> Usa teste de Nemenyi Friedman para comparação estatística
        -> Visualiza rankings e diferenças críticas entre modelos
    """

    # ===== VALIDAÇÃO DE ENTRADA =====
    # Verifica se a matriz foi passada
    if matrix is None:
        raise ValueError("Faltando uma matriz!")

    # Verifica se o dicionário de cores foi fornecido
    if colors is None:
        raise ValueError(
            "Falha ao obter um dicionário com as cores para o Diagrama de Diferença Crítica"
        )

    # ===== CÁLCULO DE RANKS =====
    # Calcula o ranking médio de cada modelo
    # ascending=False: melhor Exatidão = rank menor
    ranks = matrix.rank(axis=1, ascending=False).mean()

    # ===== TESTE ESTATÍSTICO =====
    # Realiza o teste post-hoc de Nemenyi (compara modelos múltiplos)
    # Retorna matrix com valores p (p-values) para comparações pairwise
    nemenyi = sp.posthoc_nemenyi_friedman(matrix)

    # ===== CONFIGURAÇÃO VISUAL =====
    # Configuração dos marcadores dos pontos no diagrama
    marker = {"marker": "o", "linewidth": 1}
    # Configuração dos rótulos das classes
    labelProps = {"backgroundcolor": "#ADD5F7", "verticalalignment": "top"}

    # ===== PLOTAGEM =====
    # Plota o Diagrama de Diferença Crítica usando as configurações acima
    # O diagrama mostra visualmente quais modelos são significativamente diferentes
    _ = sp.critical_difference_diagram(
        ranks,
        nemenyi,
        color_palette=colors,
        marker_props=marker,
        label_props=labelProps,
    )




# =====================================================================
# Função: plotScatterClass
# =====================================================================
def plotScatterClass(
    X: np.array,
    targets: np.array,
    xLogScale: bool = False,
    algorithm: str = "PCA",
    randomState=42,
):
    """
    # Descrição
        -> Reduz a dimensionalidade dos dados e plota um gráfico de dispersão
           para visualizar separabilidade das classes.
    
    # Parâmetros
        -> X: Array com os dados de alta dimensionalidade (features)
        -> targets: Array com os rótulos das classes
        -> xLogScale: Se True, usa escala logarítmica no eixo X
        -> algorithm: Algoritmo para redução ("PCA" ou "t-sne")
        -> randomState: Seed para reprodutibilidade
    
    # Método
        -> PCA: Redução linear rápida (bom para tendências globais)
        -> t-SNE: Redução não-linear (melhor para separabilidade local)
    """

    # ===== CODIFICAÇÃO DE RÓTULOS =====
    # Converte rótulos de texto em números para plotagem
    encoder = LabelEncoder()
    targetsEncoded = encoder.fit_transform(targets)

    # ===== REDUÇÃO DE DIMENSIONALIDADE =====
    # Reduz dados para 2 dimensões para visualização
    if algorithm == "PCA":
        # PCA: Linear, rápido, bom para visualizar estrutura global
        X_embedded = PCA(n_components=2, random_state=randomState).fit_transform(X)

    elif algorithm == "t-sne":
        # t-SNE: Não-linear, lento mas melhor para separabilidade local
        # Primeiro, reduz para ~10 dimensões se necessário (melhora performance)
        if X.shape[1] > 10:
            X = PCA(n_components=10, random_state=randomState).fit_transform(X)

        X_embedded = TSNE(
            n_components=2,
            learning_rate="auto",  # Aprendizado adaptativo
            init="pca",             # Inicialização com PCA
            perplexity=30,          # Equilíbrio entre estrutura local/global
            random_state=randomState,
            n_jobs=-1,              # Usa todos os processadores
        ).fit_transform(X)
    else:
        raise ValueError(f"Algoritmo inválido {algorithm}")

    # ===== NORMALIZAÇÃO =====
    # Escala os dados embarcados usando escalar robusto (resistente a outliers)
    X_embedded = RobustScaler().fit_transform(X_embedded)

    # ===== CRIAÇÃO DO GRÁFICO =====
    # Cria figura com tamanho apropriado
    plt.figure(figsize=(8, 6))
    
    # Escolhe paleta de cores com cores distintas para cada classe
    colors = plt.cm.get_cmap("Paired", len(encoder.classes_))
    
    # Plota pontos coloridos por classe
    # X_embedded[:, 0]: primeira dimensão reduzida
    # X_embedded[:, 1]: segunda dimensão reduzida
    # c=targetsEncoded: cores baseadas na classe
    # s=10: tamanho pequeno dos pontos para visualizar densidade
    plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=targetsEncoded[:], cmap=colors, s=10
    )

    # ===== CONFIGURAÇÃO DOS EIXOS =====
    # Rótulos descritivos dos eixos
    plt.xlabel(f"{algorithm} Dimensão 1")
    plt.ylabel(f"{algorithm} Dimensão 2")  # Corrigido: era "Dimensão 1"
    plt.title(f"Gráfico {algorithm} dos Dados de Alta Dimensionalidade")

    # Aplica escala logarítmica ao eixo X se solicitado
    if xLogScale:
        plt.xscale("log")

    # ===== LEGENDA =====
    # Cria rótulos coloridos para cada classe
    legend_labels = [
        mpatches.Patch(color=colors(i), label=label)
        for i, label in enumerate(encoder.classes_)
    ]
    plt.legend(handles=legend_labels, title="Classe")

