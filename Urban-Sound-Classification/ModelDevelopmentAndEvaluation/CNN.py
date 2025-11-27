from typing import Tuple

import numpy as np

import pandas as pd

from tensorflow import keras

from tensorflow.keras.layers import Input, BatchNormalization, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, LeakyReLU, SpatialDropout2D, GlobalAveragePooling2D  # type: ignore


class CNN(keras.Model):
    """
    Classe CNN - Rede Neural Convolucional para Classificação de Som Urbano.
    
    Esta classe implementa várias arquiteturas de Redes Neurais Convolucionais (CNN) 2D
    para classificação de características de áudio (espectrogramas, MFCCs, etc).
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Inicializa o modelo CNN."""
        super().__init__(*args, **kwargs)

    # Arquitetura utilizada no artigo "An Analysis of Audio Classification Techniques using Deep Learning Architectures"
    def create2DCNN(
        self, input_shape: Tuple, testNumber: int, numClasses: int = 10
    ) -> keras.Sequential:
        """
        Cria diferentes arquiteturas de CNN 2D baseadas no número de teste.
        
        Args:
            input_shape: Forma do tensor de entrada (altura, largura, canais)
            testNumber: Número da versão de arquitetura (1-5)
            numClasses: Número de classes para a saída (padrão: 10)
            
        Returns:
            Modelo keras.Sequential com a arquitetura CNN correspondente
        """
        
        if testNumber == 1:
            # Versão 1: Arquitetura básica com 3 blocos convolucionais
            # - 3 camadas Conv2D progressivamente maiores (32, 64, 128 filtros)
            # - Normalização de lote em cada bloco para estabilização de treinamento
            # - MaxPooling para redução dimensional e extração de características principais
            # - Camada densa com dropout para regularização e evitar overfitting
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    # Primeiro bloco convolucional - 32 filtros
                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Segundo bloco convolucional - 64 filtros
                    Conv2D(64, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Terceiro bloco convolucional - 128 filtros
                    Conv2D(128, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Camadas totalmente conectadas
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
            
        elif testNumber == 2:
            # Versão 2: Arquitetura simplificada com 2 blocos convolucionais
            # - 2 camadas Conv2D com mesma profundidade (32 filtros)
            # - Arquitetura mais leve para datasets menores
            # - Reduz tempo de treinamento e requisitos computacionais
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    # Primeiro bloco convolucional - 32 filtros
                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Segundo bloco convolucional - 32 filtros
                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Camadas totalmente conectadas
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
            
        elif testNumber == 3:
            # Versão 3: Arquitetura minimal com 1 bloco convolucional
            # - Apenas 1 camada Conv2D com 32 filtros
            # - Ideal para datasets pequenos ou prototipagem rápida
            # - Menor complexidade computacional
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    # Único bloco convolucional - 32 filtros
                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Camadas totalmente conectadas
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
            
        elif testNumber == 4:
            # Versão 4: Arquitetura profunda similar à versão 1 com dropout aumentado
            # - 3 blocos convolucionais (32, 64, 128 filtros)
            # - Dropout aumentado (0.6) para maior regularização
            # - Útil para reduzir overfitting em datasets pequenos
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    # Primeiro bloco convolucional - 32 filtros
                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Segundo bloco convolucional - 64 filtros
                    Conv2D(64, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Terceiro bloco convolucional - 128 filtros
                    Conv2D(128, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    # Camadas totalmente conectadas
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.6),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )


        elif testNumber == 5:
            # Versão 5: Arquitetura proposta avançada para melhorar generalização
            # - Blocos convolucionais mais profundos com LeakyReLU (ativa neurônios com gradiente pequeno)
            # - SpatialDropout2D para regularização espacial em mapas de características
            # - GlobalAveragePooling2D para reduzir overfitting e diminuir número de parâmetros
            # - Camada Dense maior antes da saída com dropout para melhor aprendizado
            # - Design baseado em best practices de Deep Learning moderna
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    # Primeiro bloco convolucional profundo - 32 filtros com 2 camadas
                    Conv2D(32, (3, 3), padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    Conv2D(32, (3, 3), padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    MaxPooling2D((2, 2)),
                    SpatialDropout2D(rate=0.1),

                    # Segundo bloco convolucional profundo - 64 filtros com 2 camadas
                    Conv2D(64, (3, 3), padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    Conv2D(64, (3, 3), padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    MaxPooling2D((2, 2)),
                    SpatialDropout2D(rate=0.15),

                    # Terceiro bloco convolucional profundo - 128 filtros com 2 camadas
                    Conv2D(128, (3, 3), padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    Conv2D(128, (3, 3), padding="same"),
                    BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    MaxPooling2D((2, 2)),
                    SpatialDropout2D(rate=0.2),

                    # Camadas totalmente conectadas com pooling global
                    GlobalAveragePooling2D(),
                    Dense(256, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
