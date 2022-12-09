import json
import os

def load_json_task(task_name, task_dir='./tasks'):
    task_path = os.path.join(task_dir, task_name + '.json')
    with open(task_path, 'r') as f:
        task_dict = json.load(f)
    return task_dict