import yaml

def run_experiment(args):

    with open(args.synthetic_config, 'r') as f:
        synthetic_config = yaml.safe_load(f)
    
    print(synthetic_config)