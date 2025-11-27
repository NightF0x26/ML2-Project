import json


# =====================================================================
# jsonFileManagement_pt.py
# =====================================================================
# Utilitários para serializar e desserializar dicionários Python em/desde
# arquivos JSON. Mantém as funções simples e seguras, com validações
# e tratamento básico de exceções (comentado abaixo quando aplicável).


def dictToJsonFile(dictionary: dict, filePath: str) -> None:
    """
    Descrição:
        Converte um dicionário Python em um arquivo JSON no caminho fornecido.

    Parâmetros:
        -> dictionary: Dicionário Python a ser salvo como JSON.
        -> filePath: Caminho completo (incluindo nome do arquivo) para salvar o JSON.

    Retorno:
        -> None. Em caso de erro, a função retorna a exceção (comportamento
           mantido para compatibilidade com o código original).

    Observações:
        - A função valida entradas nulas e lança `ValueError` se algum parâmetro
          essencial estiver ausente.
        - Usa `indent=4` para gerar arquivos legíveis por humanos.
    """

    # Validações de entrada
    if dictionary is None:
        raise ValueError("Nenhum dicionário foi fornecido!")

    if filePath is None:
        raise ValueError("Nenhum caminho de arquivo foi fornecido!")

    # Salva o dicionário em um arquivo JSON com indentação para leitura
    try:
        with open(filePath, "w", encoding="utf-8") as json_file:
            json.dump(dictionary, json_file, indent=4, ensure_ascii=False)

    except Exception as e:
        # Comportamento original: retornar a exceção em vez de relançar.
        # Comentário: em muitos projetos é preferível lançar a exceção para
        # ser tratada pelo chamador, mas aqui preservamos o comportamento
        # original e adicionamos a documentação desse retorno.
        return e


def jsonFileToDict(filePath: str) -> dict:
    """
    Descrição:
        Carrega um arquivo JSON e converte seu conteúdo para um dicionário Python.

    Parâmetros:
        -> filePath: Caminho completo (incluindo nome do arquivo) do JSON a ser lido.

    Retorno:
        -> dict: O dicionário carregado do arquivo JSON em caso de sucesso.
        -> None: Retorna None em caso de erro ao carregar o arquivo (comportamento
                 mantido do código original).

    Observações:
        - Valida o parâmetro `filePath` e lança `ValueError` caso esteja ausente.
        - Usa leitura com encoding UTF-8 para compatibilidade com caracteres
          acentuados (pt-BR).
    """

    # Validação de entrada
    if filePath is None:
        raise ValueError("Nenhum caminho de arquivo json foi fornecido!")

    # Tenta abrir e carregar o arquivo JSON
    try:
        with open(filePath, "r", encoding="utf-8") as json_file:
            dictionary = json.load(json_file)

        return dictionary

    except Exception:
        # Retorna None em caso de falha, mantendo compatibilidade com a
        # implementação anterior. Para debug, o chamador pode verificar
        # se o retorno é None e então tratar o erro adequadamente.
        return None

