import pandas as pd
import numpy as np
import pickle
from irec.environment.loader.full_data import FullData
from api.data.newSynthetic import NewSyntheticBanditDataset
from obp.dataset.synthetic import * 

def get_class(class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        raise Exception(f"Class not found: {class_name}.")

def create_synthetic_data(synthetic_config):
    
    obp_args = synthetic_config['obp_args']

    if obp_args.get('reward_function',None) != None:
        obp_args['reward_function'] = get_class(obp_args['reward_function'])

    if obp_args.get('behavior_function',None) != None:
        obp_args['behavior_function'] = get_class(obp_args['behavior_function'])

#---------------------------------------------------SYNTHETIC-DATA-------------------------------------------------#
    dataset = NewSyntheticBanditDataset(
        **(obp_args)
    )

    if synthetic_config['extra_args']['reward_type'] == "continuos":
        dataset.reward_min = synthetic_config['extra_args'].get('min_reward', dataset.reward_min)
        dataset.reward_max = synthetic_config['extra_args'].get('max_reward', dataset.reward_max)

    bandit_data = dataset.obtain_batch_bandit_feedback(obp_args['n_rounds'])

#-----------------------------------------------------TO-CSV-------------------------------------------------------#
    df = pd.DataFrame({
        'user_id': bandit_data['users'],
        'item_id': bandit_data['action'],
        'rating': bandit_data['reward'],
        'timestamp': range(len(bandit_data['action']))
    })
    df.to_csv('temp/bandit_data.csv', index=False)


#---------------------------------------------------SPLITTING------------------------------------------------------#
    dataset = {
        'path': "temp/bandit_data.csv",
        'random_seed': 0,
        'file_delimiter': ",",
        'skip_head': True
    }

    loader = FullData(dataset, synthetic_config['splitting'])
    train_dataset, test_dataset, _, _, test_uids, threshold = loader.process()

    test_data = bandit_data[(bandit_data['user_id'].isin(test_uids))].groupby('user_id').head(100)
    train_data = bandit_data[not (bandit_data['user_id'].isin(test_uids)) & bandit_data['timestamp'] <= threshold]
    train_idx = train_data.index.tolist()
    test_idx = test_data.index.tolist()

    train_dict = {k: bandit_data[k][train_idx] for k,v in bandit_data.items() if isinstance(v, np.array)}
    test_dict = {k: bandit_data[k][test_idx] for k,v in bandit_data.items() if isinstance(v, np.array)}

    train_dict.update({k: bandit_data[k] for k,v in bandit_data.items() if isinstance(v, int)})
    test_dict.update({k: bandit_data[k] for k,v in bandit_data.items() if isinstance(v, int)})

    train_dict['n_rounds'] = len(train_idx)
    test_dict['n_rounds'] = len(test_idx)

#----------------------------------------------------SAVE----------------------------------------------------------#
    with open("train_data.obj", "wb") as f:
        pickle.dump(train_dict, f)
    with open("test_data.obj", "wb") as f:
        pickle.dump(test_dict, f)