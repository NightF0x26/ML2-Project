from typing import Tuple
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization # type: ignore
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.layers import Add, ReLU, Dense, Dropout # type: ignore
from tensorflow.keras.utils import register_keras_serializable # type: ignore


@register_keras_serializable()
class ResidualBlock(Layer):
    """Bloco residual para rede ResNet com skip connections."""
    
    def __init__(self, filters:int, kernel_size:int=3, stride:int=1, **kwargs):
        """
        Inicializa o Bloco Residual.
        
        Args:
            filters: Número de filtros convolucionais
            kernel_size: Tamanho do kernel (padrão: 3)
            stride: Tamanho do passo para convolução (padrão: 1)
            **kwargs: Argumentos adicionais da classe Layer
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.stride = stride
        self.kernel_size = kernel_size
        self.filters = filters

        # Primeira Camada Convolucional com Normalização de Lote e ReLU
        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
        )
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        # Segunda Camada Convolucional
        self.conv2 = Conv2D(
            filters=filters, kernel_size=3, padding="same", strides=stride
        )
        self.bn2 = BatchNormalization()

        # Define a conexão de salto (skip connection)
        self.skip_connection = None

        self.add = Add()
        self.relu2 = ReLU()

    def build(self, input_shape:Tuple) -> None:
        """Constrói a conexão de salto baseada na forma de entrada."""
        # Define a conexão de salto
        if self.stride != 1 or input_shape[-1] != self.filters:
            self.skip_connection = keras.Sequential(
                [
                    Conv2D(
                        filters=self.filters,
                        kernel_size=1,
                        strides=self.stride**2,
                        padding="same",
                        use_bias=False,
                    ),
                    BatchNormalization(),
                ]
            )
        else:
            # Função identidade
            self.skip_connection = lambda x, training: x

    def call(self, x: tf.Tensor, training=False):
        """Chama o bloco residual passando pelos convoluções e adicionando a conexão de salto."""
        # Salva x para a conexão de salto
        residue = self.skip_connection(x, training=training)

        # Passa o vetor x pelas camadas previamente definidas
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Converge o resultado das camadas anteriores para o resíduo inicial
        x = self.add([x, residue])
        x = self.relu2(x)

        return x

    def get_config(self) -> dict:
        """Retorna configuração do bloco para serialização."""
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Cria bloco a partir da configuração salva."""
        return cls(**config)

class ResNet(tf.keras.Model):
    """Rede Neural Convolucional ResNet para classificação de som urbano."""
    
    def __init__(self, input_shape:Tuple, num_classes:int=10) -> None:
        """
        Inicializa o modelo ResNet.
        
        Args:
            input_shape: Forma da entrada (altura, largura, canais)
            num_classes: Número de classes de saída (padrão: 10)
        """
        super().__init__()

        self.inputLayer = Input(shape=input_shape)

        # Camadas iniciais convolucionais
        self.conv = Conv2D(
            filters=64, kernel_size=7, strides=2, padding="same", use_bias=False
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.pool = MaxPooling2D(pool_size=3, strides=2, padding="same")

        # Define múltiplos blocos de camadas
        self.layer1 = self._buildLayer(filters=64, blocks=2, stride=1)
        self.layer2 = self._buildLayer(filters=128, blocks=2, stride=2)
        self.layer3 = self._buildLayer(filters=256, blocks=2, stride=2)
        self.layer4 = self._buildLayer(filters=512, blocks=2, stride=2)

        # Camadas finais
        self.globalAvgPool = GlobalAveragePooling2D()
        self.fullyConnectedLayer = Dense(num_classes, activation="softmax")

    def _buildLayer(self, filters, blocks, stride) -> keras.Sequential:
        """
        Constrói uma camada com múltiplos blocos residuais.
        
        Args:
            filters: Número de filtros
            blocks: Número de blocos residuais
            stride: Tamanho do passo do primeiro bloco
            
        Returns:
            Camada Sequential com blocos residuais
        """
        # Cria uma lista para as camadas residuais
        residualLayers = []

        # Cria e adiciona o bloco residual inicial
        residualLayers.append(ResidualBlock(filters=filters, stride=stride))

        # Obtém todos os blocos residuais restantes
        for _ in range(1, blocks):
            residualLayers.append(ResidualBlock(filters=filters, stride=1))

        # Converte as camadas residuais em um Modelo Sequential
        return keras.Sequential(residualLayers)

    def createResNet(self, testNumber:int) -> keras.Sequential:
        """
        Cria diferentes arquiteturas ResNet baseadas no número de teste.
        
        Args:
            testNumber: Número da versão de arquitetura (1-3)
            
        Returns:
            Modelo Keras Sequential com a arquitetura ResNet
        """
        if testNumber == 1:
            # Versão 1: ResNet padrão sem dropout
            return keras.Sequential([
                # Camadas iniciais
                self.inputLayer,
                self.conv,
                self.bn,
                self.relu,
                self.pool,

                # Blocos residuais
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,

                # Aplicar pooling global e camada totalmente conectada
                self.globalAvgPool,
                self.fullyConnectedLayer
            ],name=f"ResNet-V{testNumber}")
            
        elif testNumber == 2:
            # Versão 2: ResNet com dropout adicional para regularização
            return keras.Sequential([
                # Camadas iniciais
                self.inputLayer,
                self.conv,
                self.bn,
                self.relu,
                self.pool,

                # Blocos residuais com dropout para evitar overfitting
                self.layer1,
                Dropout(0.5),
                self.layer2,
                Dropout(0.5),
                self.layer3,
                Dropout(0.5),
                self.layer4,
                Dropout(0.5),

                # Aplicar pooling global e camada totalmente conectada
                self.globalAvgPool,
                self.fullyConnectedLayer
            ], name=f"ResNet-V{testNumber}")
            
        elif testNumber == 3:
            # Versão 3: ResNet simplificada com apenas primeira e segunda camada
            return keras.Sequential([
                # Camadas iniciais
                self.inputLayer,
                self.conv,
                self.bn,
                self.relu,
                self.pool,

                # Blocos residuais limitados com dropout
                self.layer1,
                Dropout(0.5),

                # Aplicar pooling global e camada totalmente conectada
                self.globalAvgPool,
                self.fullyConnectedLayer
            ], name=f"ResNet-V{testNumber}")
