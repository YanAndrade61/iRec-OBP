import yaml
from obp.dataset import *

def get_class(class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        raise Exception(f"Class not found: {class_name}.")

if __name__ == '__main__':
    with open('examples/yaml/synthetic.yaml', 'r') as f:
        synthetic_config = yaml.load(f, Loader=yaml.BaseLoader)
    print(synthetic_config)
    print(get_class(synthetic_config['reward_function']))
    print(get_class(synthetic_config['behavior_function']))