import pandas as pd
import numpy as np
import pickle
from irec.environment.loader.full_data import FullData
from .newSynthetic import NewSyntheticBanditDataset
from obp.dataset.synthetic import * 
from .utils import check_args
import json

def get_class(class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        raise Exception(f"Class not found: {class_name}.")

def create_synthetic_data(config):

    obp_args: dict = config['obp_args']
    extra_args = config['extra_args']
    # print(json.dumps(config,indent=4))
    check_args(config)

    if obp_args.get('reward_function', None) is not None:
        obp_args['reward_function'] = get_class(obp_args['reward_function'])

    if obp_args.get('behavior_function', None) is not None:
        obp_args['behavior_function'] = get_class(obp_args['behavior_function'])

    # Create synthetic dataset
    n_rounds = obp_args.pop('n_rounds')
    synthetic_dataset = NewSyntheticBanditDataset(**obp_args)
    synthetic_dataset.user_context_file = extra_args['user_context_file']
    
    # Set reward range if reward type is continuous
    if obp_args['reward_type'] == "continuous":
        synthetic_dataset.reward_min = extra_args['min_reward'] or synthetic_dataset.reward_min
        synthetic_dataset.reward_max = extra_args['max_reward'] or synthetic_dataset.reward_max

    # Obtain batch bandit feedback
    bandit_data = synthetic_dataset.obtain_batch_bandit_feedback(n_rounds)

    # Convert data to pandas dataframe and save to csv
    df = pd.DataFrame({
        'user_id': bandit_data['users'],
        'item_id': bandit_data['action'],
        'rating': bandit_data['reward'],
        'timestamp': range(len(bandit_data['action']))
    })
    df.to_csv('.local/obp_cache/bandit_data.csv', index=False)

    loader = FullData(config['dataset'], config['splitting'])
    train_dataset, test_dataset, _, _ = loader.process()
    test_user_ids = np.unique(test_dataset.data[:, 0])
    threshold = test_dataset.data.min(axis=0)[3]

    # Get test data
    test_data =df[df['user_id'].isin(test_user_ids)].groupby('user_id').head(100)

    # Get training data
    train_data =df[df['user_id'].isin(test_user_ids) & df['timestamp'] <= threshold]
    train_indices = train_data.index.tolist()
    test_indices = test_data.index.tolist()

    #Splitting BanditFeedback
    train_dict = {k: bandit_data[k][train_indices] for k, v in bandit_data.items() if isinstance(v, list)}
    test_dict = {k: bandit_data[k][test_indices] for k, v in bandit_data.items() if isinstance(v, list)}

    train_dict.update({k: bandit_data[k] for k, v in bandit_data.items() if isinstance(v, int)})
    test_dict.update({k: bandit_data[k] for k, v in bandit_data.items() if isinstance(v, int)})
    test_dict['position'] = None

    train_dict['n_rounds'] = len(train_indices)
    test_dict['n_rounds'] = len(test_indices)

    return train_dict, test_dict