#!/usr/bin/env python3
"""Execute Jupyter notebooks to generate figures"""

import sys
import subprocess
import os

# Change to notebooks directory
notebooks_dir = '/workspace/notebooks'
os.chdir(notebooks_dir)

notebooks = [
    'Maule SCM.ipynb',
    'Canterbury SCM.ipynb'
]

for nb in notebooks:
    print(f'\n=== Executing {nb} ===\n')
    try:
        # Use jupyter execute instead of nbconvert
        result = subprocess.run(
            ['/home/ubuntu/.local/bin/jupyter', 'execute', nb],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f'Error executing {nb}:')
            print(result.stderr)
        else:
            print(f'Successfully executed {nb}')
    except Exception as e:
        print(f'Exception executing {nb}: {e}')
        continue

print('\n=== All notebooks executed ===')
