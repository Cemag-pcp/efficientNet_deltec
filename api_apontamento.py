import requests

def finalizar_cambao(classe):
    
    """
    Envia uma requisição POST para finalizar o cambão.
    """

    url = 'https://apontamentousinagem.onrender.com/pintura/api/finalizar-cambao/'
    payload = {
        'cambao_nome': classe,
        'operador': 125
    }

    response = requests.post(url, json=payload)

    print("Status:", response.status_code)
    print("Conteúdo:", response.text)
