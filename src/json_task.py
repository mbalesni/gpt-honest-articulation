import json
import os

def load_json_task(task_name, task_dir='./tasks'):
    task_path = os.path.join(task_dir, task_name + '.json')
    with open(task_path, 'r') as f:
        task_dict = json.load(f)
    return task_dict

def make_few_shots(task, num_few_shots=None):
    few_shots = task['few_shots'][:num_few_shots]
    few_shots = [task['question_prefix'] + ex['text'] + task['question_postfix'] + task['question_prompt'] + ' ' + str(ex['label']) + '.\n\n' for ex in few_shots]
    few_shots = ''.join(few_shots)
    return few_shots