#!/usr/bin/env python3

import subprocess

COMMON_ANSATZES = ['default', 'ferminet', 'deeperwin', 'psiformer']
COMMON_TASKS = ['train', 'evaluate']

for ansatz in COMMON_ANSATZES:
    for task in COMMON_TASKS:
        print(f'Running config task={task} ansatz={ansatz}...')
        command = [
            'deepqmc',
            f'task={task}',
            f'hydra.run.dir=common_config_runs/{ansatz}_{task}',
            *(
                [f'task.restdir=common_config_runs/{ansatz}_train/training']
                if task == 'evaluate'
                else [f'ansatz={ansatz}', 'task.pretrain_steps=10', 'task.steps=10']
            ),
        ]
        try:
            result = subprocess.run(command, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Test task={task} ansatz={ansatz} failed!')
            print('Job stdout:')
            print(e.stdout.decode())
            print('=========================================')
            print('Job stderr:')
            print(e.stderr.decode())
            print('=========================================')
