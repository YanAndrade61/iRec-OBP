import os

lst = [1,2,3]
os.system("models=1;")
os.system('echo $models')

"""
1. Dataset
- Criar dataset
- Mover dataset pra irec-cmdline/data/dataset/
- Atualizar o yaml dataset_loaders.yaml
- Gerar dataset pelo irec generate_dataset.py

2. Experimental
- Pegar diretorio de trabalho atual

cd $app_path/scripts/agents

python3 run_agent_best.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}"

cd $app_path/scripts/evaluation

python3 eval_agent_best.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}"
python3 print_latex_table_results.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}"

"""

