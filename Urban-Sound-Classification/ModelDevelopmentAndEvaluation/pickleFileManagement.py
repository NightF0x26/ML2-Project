# Módulo para gerenciar serialização e desserialização de objetos Python usando pickle
import pickle

def saveObject(objectObtained:object=None, filePath:str=None) -> None:
    """
    # Descrição
        -> Salva o melhor estimador encontrado (passado para a função).
    -----------------------------------------------------------------    
    Parâmetros:
    := param: objectObtained - Objeto para armazenar em um ficheiro pickle.
    := param: filePath - Caminho do ficheiro para o estimador.
    := retorno: None, pois estamos apenas salvando um estimador.
    """

    # Verifica se um estimador foi fornecido
    if objectObtained is None:
        raise ValueError("Faltando uma instância de objeto para salvar!")
    
    # Verifica se o caminho é válido
    if filePath is None:
        raise ValueError("Caminho inválido fornecido!")

    # Salva o melhor estimador em formato binário usando pickle
    with open(filePath, 'wb') as f:
        pickle.dump(objectObtained, f)

def loadObject(filePath:str=None) -> object:
    """
    # Descrição
        -> Carrega um objeto previamente salvo.
    ----------------------------------------------
    := param: filePath - Caminho do ficheiro do objeto salvo.
    := retorno: O objeto armazenado.
    """

    # Verifica se o caminho é válido
    if filePath is None:
        raise ValueError("Caminho inválido fornecido!")

    # Carrega o objeto do ficheiro em formato binário usando pickle
    with open(filePath, 'rb') as f:
        objectObtained = pickle.load(f)

    # Retorna o objeto carregado
    return objectObtained
