import pandas as pd
import numpy as np
import random
from sklearn.utils import check_random_state
from tqdm import tqdm

def select_context(n_rounds: int, n_actions: int, context_file_path: str):
    """
    Seleciona o contexto dos usuários para as rodadas e ações especificadas.

    Args:
        n_rounds (int): O número de rodadas.
        n_actions (int): O número de ações.
        context_file_path (str): O caminho do arquivo de contexto.

    Returns:
        Um tuple contendo um array de IDs de contexto e um array de vetores de contexto para cada usuário.
    """
    print("Selecionando contexto...")

    # Carrega o arquivo de contexto
    df = pd.read_csv(context_file_path, delimiter='|', converters={'context': eval})
    
    # Cria uma lista de contexto para cada usuário com base na frequência de ocorrência
    u_contexts = [list(context) for context, freq in zip(df['context'], df['freq']) for i in range(freq)]
    
    # Embaralha as listas de contexto e IDs
    u_zipped = list(zip(df['user_id'], u_contexts))
    random.shuffle(u_zipped)
    u_ids, u_contexts = zip(*u_zipped)

    print("Seleção de contexto concluída.")

    return np.array(u_ids), np.array(u_contexts)

def sample_action_context(action_dist: np.ndarray, users: np.ndarray, random_state: int = None) -> np.ndarray:
    """
    Samples actions for each user according to a distribution.

    Parameters:
        action_dist (numpy.ndarray): distribuição de probabilidade das ações para cada usuário a serem selecionadas.
        users (numpy.ndarray): Contem o indice dos usuarios a serem selecionados.
        random_state (int, optional): Semente para o gerador de números aleatórios.

    Returns:
        numpy.ndarray: Contem as ações selecionadas para cada usuário.

    """
    random_ = check_random_state(random_state)
    n_users, n_actions = action_dist.shape
    chosen_actions = np.zeros(n_users, dtype=np.int)

    cum_action_dist = np.cumsum(action_dist, axis=1)
    uniform_rvs = random_.uniform(size=n_users)

    for i in tqdm(range(n_users), desc="Selecting actions"):
        hist = set()
        for _ in range(n_actions):
            action = np.argmax(cum_action_dist[i] > uniform_rvs[i])
            if action not in hist:
                chosen_actions[i] = action
                hist.add(action)
                break
            cum_action_dist[i][action] = -1

    return chosen_actions

def check_args(config: dict) -> None:
    """
    Check whether the arguments passed in the configuration dictionary are valid.

    Args:
        config (dict): A dictionary with configurations about OBP.

    Raises:
        ValueError: If n_rounds is not equal to the sum of frequency in user_context_file, 
        or if any of the frequencies in user_context_file is greater than n_actions.
    """
    obp_args = config.get('obp_args', {})
    n_rounds = obp_args.get('n_rounds')
    n_actions = obp_args.get('n_actions')
    context_file_path = config.get('extra_args', {}).get('context_file_path')
    
    if not all([n_rounds, n_actions, context_file_path]):
        raise ValueError('Mandatory parameters were not specified: n_rounds, n_actions, context_file_path')
    
    df = pd.read_csv(context_file_path, delimiter='|', converters={'context': eval})
    
    if df['freq'].sum() != n_rounds:
        raise ValueError("The value of n_rounds must be equal to the sum of frequency in user_context_file.")
    
    if (df['freq'] > n_actions).any():
        raise ValueError("All frequency values in user_context_file must be smaller or equal to the value of n_actions.") 