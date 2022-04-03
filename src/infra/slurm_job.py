"""
Must be launched from the released project dir
"""

import os
import time
import random
import subprocess
from shutil import copyfile

import hydra
from omegaconf import DictConfig

# Unfortunately, (AFAIK) we cannot pass arguments normally (to parse them with argparse)
# that's why we are reading them from env
SLURM_JOB_ID = os.getenv('SLURM_JOB_ID')
project_dir = os.getenv('project_dir')
python_bin = os.getenv('python_bin')

# Printing the environment
print('PROJECT DIR:', project_dir)
print(f'SLURM_JOB_ID: {SLURM_JOB_ID}')
print('HOSTNAME:', subprocess.run(['hostname'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
print(subprocess.run([os.path.join(os.path.dirname(python_bin), 'gpustat')], stdout=subprocess.PIPE).stdout.decode('utf-8'))

@hydra.main(config_name=os.path.join(project_dir, 'experiment_config.yaml'))
def main(cfg: DictConfig):
    os.chdir(project_dir)

    target_data_dir_base = os.path.dirname(cfg.dataset.path)
    if os.path.islink(target_data_dir_base):
        os.makedirs(os.readlink(target_data_dir_base), exist_ok=True)
    else:
        os.makedirs(target_data_dir_base, exist_ok=True)

    copyfile(cfg.dataset.path_for_slurm_job, cfg.dataset.path)
    print(f'Copied the data: {cfg.dataset.path_for_slurm_job} => {cfg.dataset.path}. Starting the training...')

    training_cmd = open('training_cmd.sh').read()
    print('<=== TRAINING COMMAND ===>')
    print(training_cmd)
    os.system(training_cmd)


if __name__ == "__main__":
    main()
