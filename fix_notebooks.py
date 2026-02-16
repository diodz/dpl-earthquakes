#!/usr/bin/env python3
"""Fix notebook output cells that are missing metadata."""
import json
import sys

def fix_notebook(notebook_path):
    """Add empty metadata to output cells that are missing it."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        for output in cell.get('outputs', []):
            if 'metadata' not in output:
                output['metadata'] = {}
                modified = True
            # Add execution_count for execute_result outputs
            if output.get('output_type') == 'execute_result' and 'execution_count' not in output:
                output['execution_count'] = None
                modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed {notebook_path}")
    else:
        print(f"No fixes needed for {notebook_path}")

if __name__ == '__main__':
    for nb_path in sys.argv[1:]:
        fix_notebook(nb_path)
