# Módulo para gerenciar dados do UrbanSound8k e realizar validação cruzada
from typing import Tuple, Callable
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Ferramentas de pré-processamento e métricas de avaliação
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from tensorflow import keras
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import get_custom_objects  # type: ignore

from .DataVisualization import plotNetworkTrainingPerformance, plotConfusionMatrix
from .pickleFileManagement import saveObject, loadObject


class UrbanSound8kManager:
    def __init__(
        self,
        featuresToUse: str = None,
        modelType: str = None,
        testNumber: int = None,
        pathsConfig: dict = None,
    ) -> None:
        """
        # Descrição
            -> Construtor que define novas instâncias da classe UrbanSound8kManager.
        ------------------------------------------------------------------------------------
        := param: featuresToUse - Características a considerar (das previamente extraídas).
        := param: modelType - Nome do modelo que será usado para treino.
        := param: testNumber - Número do teste que o modelo atual irá executar.
        := param: pathsConfig - Dicionário usado para armazenar os caminhos dos arquivos importantes do projeto.
        := return: None, pois estamos apenas instanciando a classe.
        """

        # Verifica se uma dimensionalidade dos dados foi fornecida
        if featuresToUse is None:
            raise ValueError("Faltando o valor para a dimensionalidade dos dados!")

        # Verifica se o tipo de modelo foi passado
        if modelType is None:
            raise ValueError(
                'Faltando o tipo de modelo para ser usado no treino! [Use "CNN", "MLP" ou "YAMNET" - dependendo do modelo que deseja treinar nos dados selecionados!]'
            )

        # Verifica se o número do teste foi fornecido
        if testNumber is None:
            raise ValueError("Faltando o número do teste do modelo!")

        # Verifica se a configuração de caminhos foi fornecida
        if pathsConfig is None:
            raise ValueError("Faltando o dicionário com a configuração dos caminhos!")

        # Salva a dimensionalidade dos dados
        self.featuresToUse = featuresToUse

        # Salva o tipo de modelo
        self.modelType = modelType

        # Salva o número do teste
        self.testNumber = testNumber

        # Salva o dicionário com os caminhos dos arquivos
        self.pathsConfig = pathsConfig

    def manageData(self) -> pd.DataFrame:
        """
        # Descrição
            -> Este método permite gerenciar facilmente os dados de todos os
            DataFrames coletados para criar um DataFrame com todas as informações.
        ------------------------------------------------------------------------
        := return: DataFrames de treino e teste do Pandas.
        """

        if (
            self.featuresToUse not in self.pathsConfig["Datasets"]["Fold-1"].keys()
            and self.featuresToUse != "transfer"
        ):
            # Dimensionalidade dos dados inválida
            raise ValueError(
                f'Características inválidas selecionadas! (Escolha entre {self.pathsConfig["Datasets"]["Fold-1"].keys()})'
            )

        # Cria um dataframe com todos os dados coletados dos folds (inicialmente vazio)
        df = None

        # Itera pelos folds dos datasets (tratamento especial para transfer learning)
        if self.featuresToUse == "transfer":
            df = pd.read_pickle(self.pathsConfig["Datasets"][self.featuresToUse])
        else:
            for fold in range(1, 11):
                # Carrega o dataframe do fold atual
                fold_df = pd.read_pickle(
                    self.pathsConfig["Datasets"][f"Fold-{fold}"][self.featuresToUse]
                )

                # Se o DataFrame ainda não foi criado, inicializa
                if df is None:
                    df = fold_df
                else:
                    # Concatena o DataFrame do fold atual
                    df = pd.concat([df, fold_df], axis=0, ignore_index=True)

        return df

    def getTrainTestSplitFold(
        self, testFold: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        # Descrição
            -> Este método permite obter os conjuntos de treino e teste do UrbanSound8k
            considerando o fold selecionado como teste.
        ---------------------------------------------------------------------------------------------------------------------
        := param: testFold - Fold do conjunto de dados a ser usado para teste. [NOTA] testFold deve estar em [1, 2, ..., 10].
        := return: Conjuntos de treino e teste para realizar a validação cruzada 10-Fold
        """

        # Verifica se o testFold foi fornecido
        if testFold is None:
            raise ValueError("Faltando o número do Fold de Teste!")

        # Verifica a integridade do fold selecionado
        if testFold < 1 or testFold > 10:
            raise ValueError("Fold de Teste inválido!")

        # Gerencia os dados de todos os DataFrames coletados
        df = self.manageData()

        # Calcula a quantidade de rótulos alvo únicos
        numClasses = np.unique(df["target"]).size

        # Separa os dados em treino, validação e teste
        train_df = df[(df["fold"] != testFold) & (df["fold"] != (testFold % 10 + 1))]
        validation_df = df[(df["fold"] == (testFold % 10 + 1))]
        test_df = df[(df["fold"] == testFold)]

        # Reseta os índices
        train_df = train_df.reset_index(drop=True)
        validation_df = validation_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Binariza a coluna alvo no treino e transforma no teste
        labelBinarizer = LabelBinarizer()
        trainBinarizedTarget = labelBinarizer.fit_transform(train_df["target"])
        validationBinarizedTarget = labelBinarizer.transform(validation_df["target"])
        testBinarizedTarget = labelBinarizer.transform(test_df["target"])
        self.classes_ = labelBinarizer.classes_

        # Atualiza os DataFrames de treino, validação e teste com o alvo binarizado
        train_df = pd.concat(
            [
                train_df.drop(columns=["target"]),
                pd.DataFrame(trainBinarizedTarget, columns=labelBinarizer.classes_),
            ],
            axis=1,
        )
        validation_df = pd.concat(
            [
                validation_df.drop(columns=["target"]),
                pd.DataFrame(
                    validationBinarizedTarget, columns=labelBinarizer.classes_
                ),
            ],
            axis=1,
        )
        test_df = pd.concat(
            [
                test_df.drop(columns=["target"]),
                pd.DataFrame(testBinarizedTarget, columns=labelBinarizer.classes_),
            ],
            axis=1,
        )

        # Avalia o tipo de dimensionalidade dos dados e adapta o método de extração de características
        # Trata dados 1D processados (MFCCs ou outras características)
        if (
            self.featuresToUse == "1D-Processed-MFCCs"
            or self.featuresToUse == "1D-Processed-Features"
        ):
            # Define as colunas das características e do alvo
            featuresCols = train_df.columns[2 : len(train_df.columns) - numClasses]
            targetCols = train_df.columns[-numClasses:]

            # Separa os dados em X e y para treino, validação e teste
            X_train = train_df[featuresCols].to_numpy()
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols].to_numpy()
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols].to_numpy()
            y_test = test_df[targetCols].to_numpy()

            # Normaliza os dados usando média e desvio padrão do conjunto de treino
            # Isso garante que os dados estejam em escala padronizada
            mean = X_train.mean()
            std = X_train.std()

            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

        elif self.featuresToUse == "2D-Raw-MFCCs":
            # Trata dados 2D brutos (matrizes MFCC)
            # Define as colunas das características e do alvo
            featuresCols = "MFCC"
            targetCols = train_df.columns[-numClasses:]

            # Separa os dados em X e y para treino, validação e teste
            X_train = train_df[featuresCols]
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols]
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols]
            y_test = test_df[targetCols].to_numpy()

            # Empilha os dados
            X_train = np.stack(X_train)
            X_val = np.stack(X_val)
            X_test = np.stack(X_test)

            # Normaliza os dados 2D usando média e desvio padrão do conjunto de treino
            mean = X_train.mean()
            std = X_train.std()

            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

        elif self.featuresToUse == "transfer":
            # Trata dados de transfer learning (embeddings pré-computados)
            # Define as colunas das características e do alvo
            featuresCols = "embedding"
            targetCols = train_df.columns[-numClasses:]

            # Separa os dados em X e y para treino, validação e teste
            X_train = train_df[featuresCols]
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols]
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols]
            y_test = test_df[targetCols].to_numpy()

            X_train = np.stack(X_train)
            X_val = np.stack(X_val)
            X_test = np.stack(X_test)

        else:
            # Tratamento para tipos de dados não reconhecidos
            raise ValueError(
                "[ALGO DEU ERRADO] Dimensionalidade dos dados inválida selecionada!"
            )

        # Retorna os conjuntos computados
        return X_train, y_train, X_val, y_val, X_test, y_test

    def getAllFolds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        # Descrição
            -> Este método ajuda a obter todos os dados de todos os folds
            que serão usados para criar o gráfico t-SNE.
        -------------------------------------------------------------
        := return: Conjuntos X e y.
        """

        # Gerencia os dados
        df = self.manageData()

        # Avalia o tipo de dimensionalidade dos dados e adapta o método de extração
        # Trata dados 1D processados
        if self.featuresToUse == "1D-Processed-MFCCs" or self.featuresToUse == "1D-Processed-Features":
            # Define as colunas das características e do alvo
            featuresCols = df.columns[2:-1]
            targetCols = df.columns[-1:]

            # Separa os dados em X e y
            X = df[featuresCols].to_numpy()
            y = df[targetCols].to_numpy()

        elif self.featuresToUse == "2D-Raw-MFCCs":
            # Trata dados 2D brutos (matrizes MFCC)
            # Define as colunas das características e do alvo
            featuresCols = "MFCC"
            targetCols = df.columns[-1:]

            # Separa os dados em X e y
            X = df[featuresCols]
            y = df[targetCols].to_numpy()

            # Empilha os dados em array 3D
            X = np.stack(X)

        elif self.featuresToUse == "transfer":
            # Trata dados de transfer learning (embeddings pré-computados)
            # Define as colunas das características e do alvo
            featuresCols = "embedding"
            targetCols = "target"

            # Separa os dados em X e y
            X = df[featuresCols]
            y = df[targetCols].to_numpy()

            # Empilha os embeddings em array 3D
            X = np.stack(X)

        else:
            # Tratamento para tipos de dados não reconhecidos
            raise ValueError(
                "[ALGO DEU ERRADO] Dimensionalidade dos dados inválida selecionada!"
            )

        return X, y

    def crossValidate(
        self,
        createModel: Callable[[], keras.models.Sequential],
        numberFolds: int = 10,
        epochs: int = 100,
        batchSize: int = 32,
        callbacks=lambda: [],
    ) -> Tuple[list[float], list[History], list[np.ndarray]]:
        """
        # Descrição
            -> Este método permite realizar validação cruzada no conjunto UrbanSound8k
            dado um modelo compilado.
        ------------------------------------------------------------------------------------
        := param: compiledModel - Modelo sequencial Keras previamente compilado.
        := param: numberFolds - Número de folds para realizar a validação cruzada.
        := param: epochs - Número de períodos para treinar o modelo em cada fold.
        := param: callbacks - Lista de parâmetros para monitorar e modificar o comportamento do modelo durante treino, avaliação e inferência.
        := return: Lista com as métricas de desempenho (History) do modelo em cada fold.
        """

        assert 0 < numberFolds <= 10, f"número de iterações inválido: {numberFolds}"

        # Inicializa lista para armazenar o histórico do modelo em cada fold
        histories = []

        # Inicializa lista para armazenar as matrizes de confusão do modelo em cada fold
        confusionMatrices = []

        # Lista para armazenar as exatidão balanceadas dos folds
        foldsBalancedAccuracy = []

        # Realiza validação cruzada iterando sobre cada fold
        for testFold in range(1, numberFolds + 1):
            # Cria nova instância do modelo (uma nova instância para cada fold)
            compiledModel = createModel(testNumber=self.testNumber)

            # Particiona os dados em treino e validação
            X_train, y_train, X_val, y_val, X_test, y_test = self.getTrainTestSplitFold(
                testFold=testFold
            )

            # Obtém o caminho do modelo e do histórico do fold atual
            modelFilePath = self.pathsConfig["ModelDevelopmentAndEvaluation"][self.modelType][f"Test-{self.testNumber}"][f"Fold-{testFold}"]["Model"]
            historyFilePath = self.pathsConfig["ModelDevelopmentAndEvaluation"][self.modelType][f"Test-{self.testNumber}"][f"Fold-{testFold}"]["History"]

            # Verifica se o fold já foi computado (reutiliza modelos previamente treinados)
            foldAlreadyComputed = os.path.exists(modelFilePath)

            # Garante que o caminho do fold do modelo existe (cria diretórios se necessário)
            modelFoldPath = Path("/".join(modelFilePath.split("/")[:-1]))
            modelFoldPath.mkdir(parents=True, exist_ok=True)

            # Se não treinou o modelo, treina agora
            if not foldAlreadyComputed:
                # Treina o modelo com os dados de treino e validação
                history = compiledModel.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batchSize,
                    epochs=epochs,
                    callbacks=callbacks(),
                )

                # Salva o histórico do treinamento (perda, exatidão por Período)
                saveObject(history, filePath=historyFilePath)

                # Salva o modelo treinado
                compiledModel.save(modelFilePath)

                # Limpa a sessão do Keras para liberar memória
                keras.backend.clear_session()

            else:
                # Modelo já foi treinado, carrega os resultados anteriores
                # Carrega o histórico previamente computado
                history = loadObject(filePath=historyFilePath)

                # Carrega o modelo
                compiledModel = load_model(modelFilePath)

            # Faz previsões no conjunto de teste (converte probabilidades em classes)
            y_pred = np.argmax(compiledModel.predict(X_test), axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Calcula a exatidão balanceada do fold atual (importante para datasets desbalanceados)
            currentFoldAccuracy = balanced_accuracy_score(y_true, y_pred)

            # Adiciona a exatidão do fold à lista
            foldsBalancedAccuracy.append(currentFoldAccuracy)

            # Computa matriz de confusão para avaliar desempenho por classe
            confusionMatrix = confusion_matrix(y_true, y_pred)

            # Plota desempenho do treino e matriz de confusão do modelo atual
            plotNetworkTrainingPerformance(
                confusionMatrix=confusionMatrix,
                title=f"[Test-{self.testNumber}] [{self.modelType}] Fold-{testFold}",
                trainHistory=history.history,
                targetLabels=self.classes_,
            )

            # Adiciona resultados do fold à lista de resultados
            histories.append(history)
            confusionMatrices.append(confusionMatrix)

        # Retorna os históricos e as matrizes de confusão
        return np.array(foldsBalancedAccuracy), histories, confusionMatrices

    def plotGlobalConfusionMatrix(self, confusionMatrices: list[np.ndarray]) -> None:
        """
        # Descrição
            -> Este método ajuda a calcular e exibir a matriz de confusão global.
        ----------------------------------------------------------------------------
        := param: confusionMatrices - Lista com todas as matrizes de confusão computadas em todos os folds.
        := return: None, pois estamos apenas plotando a matriz de confusão.
        """

        # Computa a matriz de confusão global somando todas as matrizes dos folds
        # Isso fornece uma visão geral do desempenho do modelo em todos os dados
        globalConfusionMatrix = confusionMatrices[0]
        for m in confusionMatrices[1:]:
            globalConfusionMatrix += m

        # Plota a matriz de confusão global agregada
        plotConfusionMatrix(
            globalConfusionMatrix,
            title="Matriz de Confusão Global",
            targetLabels=self.classes_,
        )

