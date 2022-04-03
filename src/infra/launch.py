"""
Run a __reproducible__ experiment on __allocated__ resources
It submits a slurm job(s) with the given hyperparams which will then execute `slurm_job.py`
This is the main entry-point
"""

import os
import subprocess
import re

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.infra.utils import create_project_dir, recursive_instantiate

#----------------------------------------------------------------------------

HYDRA_ARGS = "hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled"

#----------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    recursive_instantiate(cfg)
    OmegaConf.set_struct(cfg, True)
    cfg.env.project_path = str(cfg.env.project_path) # This is needed to evaluate ${hydra:runtime.cwd}

    before_train_cmd = '\n'.join(cfg.env.before_train_commands)
    before_train_cmd = before_train_cmd + '\n' if len(before_train_cmd) > 0 else ''
    torch_extensions_dir = os.environ.get('TORCH_EXTENSIONS_DIR', cfg.env.torch_extensions_dir)
    training_cmd = f'{before_train_cmd}TORCH_EXTENSIONS_DIR={torch_extensions_dir} cd {cfg.project_release_dir} && {cfg.env.python_bin} src/train.py {HYDRA_ARGS}'
    quiet = cfg.get('quiet', False)
    training_cmd_save_path = os.path.join(cfg.project_release_dir, 'training_cmd.sh')
    cfg_save_path = os.path.join(cfg.project_release_dir, 'experiment_config.yaml')

    if not quiet:
        print('<=== TRAINING COMMAND START ===>')
        print(training_cmd)
        print('<=== TRAINING COMMAND END ===>')

    is_running_from_scratch = True

    if cfg.training.resume == "latest" and os.path.isdir(cfg.project_release_dir) and os.path.isfile(training_cmd_save_path) and os.path.isfile(cfg_save_path):
        is_running_from_scratch = False
        if not quiet:
            print("We are going to resume the training and the experiment already exists. " \
                "That's why the provided config/training_cmd are discarded and the project dir is not created.")

    if is_running_from_scratch and not cfg.print_only:
        create_project_dir(
            cfg.project_release_dir,
            cfg.env.objects_to_copy,
            cfg.env.symlinks_to_create,
            quiet=quiet,
            ignore_uncommited_changes=cfg.get('ignore_uncommited_changes', False),
            overwrite=cfg.get('overwrite', False))

        with open(training_cmd_save_path, 'w') as f:
            f.write(training_cmd + '\n')
            if not quiet:
                print(f'Saved training command in {training_cmd_save_path}')

        with open(cfg_save_path, 'w') as f:
            OmegaConf.save(config=cfg, f=f)
            if not quiet:
                print(f'Saved config in {cfg_save_path}')

    if not cfg.print_only:
        os.chdir(cfg.project_release_dir)

    if cfg.slurm:
        assert Path(cfg.dataset.path_for_slurm_job).exists()

        curr_job_id = None

        for i in range(cfg.job_sequence_length):
            if i == 0:
                deps_args_str = ''
            else:
                deps_args_str = f'--dependency=afterany:{curr_job_id}'

            # Submitting the slurm job
            qos_arg_str = f'--account {os.environ["PRIORITY_BOOST_ACC"]}' if cfg.use_qos else ''
            output_file_arg_str = f'--output {cfg.project_release_dir}/slurm_{i}.log'
            submit_job_cmd = f'sbatch {cfg.sbatch_args_str} {output_file_arg_str} {qos_arg_str} --export=ALL,{cfg.env_args_str} {deps_args_str} src/infra/slurm_job_proxy.sh'

            if cfg.print_only:
                print(submit_job_cmd)
                curr_job_id = "DUMMY_JOB_ID"
            else:
                result = subprocess.run(submit_job_cmd, stdout=subprocess.PIPE, shell=True)
                output_str = result.stdout.decode("utf-8").strip("\n") # It has a format of "Submitted batch job 17033559"
                if not quiet or i == 0:
                    print(output_str)
                curr_job_id = re.findall(r"^Submitted\ batch\ job\ \d{5,8}$", output_str)
                assert len(curr_job_id) == 1, f"Bad output: `{output_str}`"
                curr_job_id = int(curr_job_id[0][len('Submitted batch job '):])
    else:
        assert cfg.job_sequence_length == 1, "You can use a job sequence only when running via slurm."
        if cfg.print_only:
            print(training_cmd)
        else:
            os.system(training_cmd)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
