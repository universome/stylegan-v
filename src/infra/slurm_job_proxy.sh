#!/bin/bash
# We need this proxy so not to put the shebang into `slurm_job.py`
# We cannot put a shebang there since we use different python executors for it
$python_bin $python_script
