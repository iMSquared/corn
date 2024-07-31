#!/usr/bin/env python3

import subprocess
from hydra_zen import (store, zen, hydrated_dataclass)
import os
from pkm.util.path import get_path


def cleanup():
    subprocess.run(['pkill', '-f', 'from multiprocessing'])
    subprocess.run(['pkill', '-f', 'python3 sample_goal'])

@store(name="run_once")
def _main(obj:str, thin:bool=False, april:bool=True, data:bool=True):
    # WHERE SHOULD BE THE WORKING DIRECTORY FOR THE SUBPROCESSES?
    cwd = str(get_path('../../../scripts/real'))

    # 1. SAMPLE GOAL
    cleanup()
    if data:
        cmd = ['python3', 'sample_goal_from_data.py', '-obj', obj, '-rand_goal']
        if thin:
            cmd.extend(['-thin'])
        subprocess.run(cmd, cwd=cwd)
    else:
        subprocess.run(['python3', 'sample_goal_pose.py'],
                    cwd=cwd)
    
    # 2. RUN POLICY
    cleanup()
    cmd = ['python3', 'controller.py', '-ckpt', '/home/user/corn_runtime/dagger/', '-dagger',
                    '-obj', obj, '-april', str(int(april)) ]
    if thin:
        cmd.extend(['-thin'])
    subprocess.run(cmd, cwd=cwd)
    


def main():
    store.add_to_hydra_store()
    zen(_main).hydra_main(config_name='run_once',
                          version_base='1.1',
                          config_path=None)


if __name__ == '__main__':
    main()
