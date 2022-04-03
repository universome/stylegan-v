import os
import argparse
import copy
from typing import List, Dict, Optional
from omegaconf import OmegaConf, DictConfig
from src.infra.utils import cfg_to_args_str

#----------------------------------------------------------------------------

HYDRA_ARGS = "hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled"

#----------------------------------------------------------------------------

def batch_launch(launcher: str, experiments_dir: os.PathLike, cfg: DictConfig, datasets: List[str], print_only: bool, time: str, use_qos: bool=False, other_args: Dict={}, num_gpus: int=4, *args, **kwargs):
    for dataset in datasets:
        for exp_args in construct_experiments_args(cfg, *args, **kwargs):
            exp_args['sbatch_args.time'] = time
            exp_args['experiments_dir'] = experiments_dir
            exp_args['dataset'] = dataset
            exp_args['env'] = 'ibex'
            exp_args['use_qos'] = use_qos
            exp_args = {**exp_args, **other_args}
            curr_exp_args_str = cfg_to_args_str(exp_args, use_dashes=False)
            launching_command = f"{launcher} num_gpus={num_gpus} {curr_exp_args_str}"

            if print_only:
                os.makedirs(exp_args['experiments_dir'], exist_ok=True)
                print(launching_command)
            else:
                os.system(launching_command)

#----------------------------------------------------------------------------

def construct_experiments_args(cfg: DictConfig, experiments_list: Optional[List[str]]=None, suffix: str="") -> List[Dict]:
    args_dicts = []
    common_cfg = cfg.get('common_args', {})

    for exp_name, exp_cfg in to_dict(cfg.experiments).items():
        if not experiments_list is None and not exp_name in experiments_list:
            continue
        curr_exp_cfg = {**copy.deepcopy(to_dict(common_cfg)), **to_dict(exp_cfg)}
        curr_exp_cfg['exp_suffix'] = f'{exp_name}{suffix}'
        args_dicts.append(curr_exp_cfg)

    return args_dicts

#----------------------------------------------------------------------------

def to_dict(cfg) -> Dict:
    return OmegaConf.to_container(OmegaConf.create({**cfg}))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments launcher")
    parser.add_argument('-e', '--series_name', type=str, required=True, help="Which experiments series to launch?")
    parser.add_argument('-d', '--datasets', required=True, type=str, help='Comma-separate list of datasets')
    parser.add_argument('-p', '--print_only', action='store_true', help='Just print commands and exit?')
    parser.add_argument('-t', '--time', type=str, default='1-0', help='Which time to specify for the sbatch command?')
    parser.add_argument('-q', '--use_qos', action='store_true', help='Should we use QoS to launch jobs?')
    parser.add_argument('--experiments_list', type=str, help='Should we run only some specific experiments from this experiments series?')
    parser.add_argument('--other_args', type=str, default="", help='Additional arguments for the experiments')
    parser.add_argument('--suffix', type=str, default="", help='Additional suffix for the experiments')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use per each experiment')
    parser.add_argument('--project_dir', type=str, default=os.getcwd(), help='Project directory path')
    parser.add_argument('--project_dir_for_exps_cfg', type=str, help="Overwrite the project directory to use for experiments.yaml. Useful for debugging the config.")
    args = parser.parse_args()

    os.chdir(args.project_dir)
    user = os.environ.get('USER', 'unknown')
    python_bin = os.path.join(args.project_dir, 'env/bin/python')
    launcher = f"{python_bin} src/infra/launch.py {HYDRA_ARGS} +quiet=true slurm=true"
    experiments_dir = f'experiments/{user}/{args.series_name}'
    exps_cfg_path = os.path.join(args.project_dir if args.project_dir_for_exps_cfg is None else args.project_dir_for_exps_cfg, 'src/infra/experiments.yaml')
    all_exp_series = OmegaConf.load(exps_cfg_path)
    assert args.series_name in all_exp_series, f"Experiments series not found: {args.series_name}"
    cfg = all_exp_series[args.series_name]
    datasets = args.datasets.split(',')
    experiments_list = None if args.experiments_list is None else args.experiments_list.split(',')
    other_args = {kv.split('=')[0]: kv.split('=')[1] for kv in args.other_args.split(',') if len(kv.split('=')) == 2}

    batch_launch(
        launcher=launcher,
        experiments_dir=experiments_dir,
        cfg=cfg,
        datasets=datasets,
        print_only=args.print_only,
        time=args.time,
        use_qos=args.use_qos,
        experiments_list=experiments_list,
        other_args=other_args,
        suffix=args.suffix,
        num_gpus=args.num_gpus,
    )

#----------------------------------------------------------------------------
