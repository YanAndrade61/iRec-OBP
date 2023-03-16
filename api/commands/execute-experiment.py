import yaml

def execute_experiments(args):

    with open(args.synthetic_config, 'r') as f:
        synthetic_config = yaml.load(f, Loader=yaml.BaseLoader)
    
    