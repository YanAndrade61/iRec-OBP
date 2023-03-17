from api.data.newSynthetic import NewSyntheticBanditDataset
from obp.dataset.synthetic import * 

def get_class(class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        raise Exception(f"Class not found: {class_name}.")

def create_synthetic_data(synthetic_config):
    
    if synthetic_config['obp_args'].get('reward_function',None) != None:
        synthetic_config['obp_args']['reward_function'] = get_class(synthetic_config['obp_args']['reward_function'])

    if synthetic_config['obp_args'].get('behavior_function',None) != None:
        synthetic_config['obp_args']['behavior_function'] = get_class(synthetic_config['obp_args']['behavior_function'])

    dataset = NewSyntheticBanditDataset(
        **(synthetic_config['obp_args'])
    )

    if synthetic_config['extra_args']['reward_type'] == "continuos":
        dataset.reward_min = synthetic_config['extra_args']['min_reward'] or dataset.reward_min
        dataset.reward_max = synthetic_config['extra_args']['max_reward'] or dataset.reward_max