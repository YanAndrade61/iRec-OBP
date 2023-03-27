"""
    2. Experimental
    - Pegar diretorio de trabalho atual

    cd $app_path/scripts/agents

    python3 run_agent_best.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}"

    cd $app_path/scripts/evaluation

    python3 eval_agent_best.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}"
    python3 print_latex_table_results.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}"
"""
import os

def run_experimental(args):
    cwd = os.popen('pwd').read().rstrip()
    path_script = ['irec','irec-cmdline','app','scripts']
    path_script = os.path.join(cwd,*path_script)

    os.system(f"models={args.agents};echo $models")
    print(cwd)     
    print(args)

    pass


if __name__ == "__main__":
    run_experimental()