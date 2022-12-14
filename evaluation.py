import numpy as np

from src.openai_model import OpenAIGPT3
from src.json_task import load_json_task, make_few_shots
import logging
import os
import datetime

models = [
    'ada',
    'babbage',
    'curie',
    'davinci',
    'text-davinci-003',
    'code-davinci-002',
    # fine-tuned davinci
]

def classify(model, task, few_shot=True, articulation=None, max_length=20, stop_string='\n'):
    '''
    Answer a single question with a single API call.

    Parameters:
    - model: OpenAIGPT3 model
    - base_prompt: prompt to use for all questions
    - questions: list of questions to classify
    - max_length: maximum length of the output
    - stop_string: string to stop generation
    '''

    knowledge = make_few_shots(task) if few_shot else articulation
    questions = [task['question_prefix'] + question['text'] + task['question_postfix'] + task['question_prompt'] for question in task['questions']]

    common_prompt = task['instruction'].strip() + '\n\n' + knowledge.strip() + '\n\n'
    prompts = [common_prompt + question for question in questions]

    outputs = model.generate_text(prompts, max_length=task['max_length'] or max_length, stop_string=task['stop_string'] or stop_string, output_regex=r'\d+')
    return process_classifications(outputs)

def classify_batch(model, task, few_shot=True, articulation=None, max_length=250, stop_string='\n\n##', batch_size=5):
    '''
    Answer a batch of questions with a single API call.

    Parameters:
    - model: OpenAIGPT3 model
    - task: task to classify
    - few_shot: whether to use few shot examples
    - articulation: whether to use articulation
    - max_length: maximum length of the output
    - stop_string: string to stop generation
    - batch_size: number of questions to classify at once. Empirically found that 5 questions per
        prompt gives 100% similar predictions to querying the API with individual questions, tested on 
        a total of 20 code snippet classification examples, using code-davinci-002 and text-davinci-003.
    '''

    assert articulation is not None or few_shot is not None, 'Must specify either articulation or few_shot'

    knowledge = make_few_shots(task) if few_shot else articulation
    base_prompt = task['instruction'].strip() + '\n\n' + knowledge.strip() + '\n\n'

    batches = []
    for i in range(0, len(task['questions']), batch_size):
        questions = [
            task['question_prefix'].replace('Example:', 'Example ' + str((i % batch_size)+1) + ':') + question['text'] + task['question_postfix'] \
                for i, question in enumerate(task['questions'][i:i+batch_size])
        ]
        batch_questions_str = '\n'.join(questions)
        batch_prompt = base_prompt + task['question_prefix_bulk'] + batch_questions_str + task['question_prompt_bulk']
        batches.append(batch_prompt)

    batch_outputs = model.generate_text(batches, max_length=task['max_length_bulk'] or max_length, 
                                        stop_string=task['stop_string_bulk'] or stop_string, output_regex_all=True,
                                        output_regex=task['answer_regex_bulk'], output_prefix=task['question_prompt_bulk'])

    outputs = [output for batch in batch_outputs for output in batch]
    return process_classifications(outputs)

def process_classifications(outputs):
    preds = []
    for output in outputs:
        try:
            output = output.strip()
            output = int(output)
        except:
            logging.warn(f'Could not parse output "{output}" as an integer. Defaulting to -1.')
            output = -1
        preds.append(output)
    return preds

def evaluate_model_on_task(model_name, task_name, return_preds=False, 
                           few_shot=True, verbose=False, vverbose=False, 
                           bulk=True, batch_size=5, log_dir=None):
    model = OpenAIGPT3(model_name, log_dir=log_dir)
    task = load_json_task(task_name)

    if bulk:
        preds = classify_batch(model=model, task=task, few_shot=few_shot, batch_size=batch_size)
    else:
        preds = classify(model=model, task=task, few_shot=few_shot)
    
    # print(f'preds: (type: {type(preds)})')
    # print(preds)
    # print()
    questions_labels = [question['label'] for question in task['questions']]
    
    classification_log_dir = os.path.join(log_dir, 'classifications')
    os.makedirs(classification_log_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{time_str}_{model_name}.txt'
    classification_log_file = os.path.join(classification_log_dir, file_name)
    with open(classification_log_file, 'w+') as f:
        for i in range(len(task['questions'])):
            i_str = str((i+1)).zfill(len(str(len(task['questions']))))
            correct_str = '(correct)' if preds[i] == questions_labels[i] else '(wrong)'
            f.write(f"{i_str} {correct_str} (pred {preds[i]}, true {task['questions'][i]['label']}) {task['questions'][i]['text']}\n")
            if vverbose:
                print(f"{i_str} {correct_str} (pred {preds[i]}, true {task['questions'][i]['label']}) {task['questions'][i]['text']}")

    # print(f'questions_labels: (type: {type(questions_labels)})')
    # print(questions_labels)
    num_correct = np.sum(np.array(preds) == np.array(questions_labels))
    num_total = len(questions_labels)
    acc = num_correct / num_total

    if verbose or vverbose:
        print(f'accuracy: {acc * 100:.2f}% ({num_correct}/{num_total})')
    if return_preds:
        return acc, preds
    return acc

def evaluate_model_honesty_articulateness(model_name, task_name, articulation, preds_from_trained, 
                                         return_preds=False, verbose=False, vverbose=False,
                                         bulk=True, batch_size=5, log_dir=None, articulation_idx=None,
                                         articulator=None):
    '''Measure the honesty & articulateness score (HA score). Computed as the percentage of examples
    where, given only the model's explanation of its classification algorithm,
    another model predicts the same answer, without few-shot examples or fine-tuning.

    In terms of the Critiques paper (https://arxiv.org/pdf/2206.05802.pdf), it 
    is an inverse of discriminator-critique (DC) gap with three differences: 
    - it is computed on a simple classification task
    - instead of critiquing, a model explains how it solves the task
    - the explanation is given on a task level rather than for each example.
    '''

    model = OpenAIGPT3(model_name, log_dir=log_dir)
    task = load_json_task(task_name)

    if bulk:
        preds_from_articulations = classify_batch(model, task, few_shot=False, articulation=articulation, max_length=task['max_length'], stop_string=task['stop_string_bulk'], batch_size=batch_size)
    else:
        preds_from_articulations = classify(model, task, few_shot=False, articulation=articulation, max_length=task['max_length'], stop_string=task['stop_string'])
    
    classification_log_dir = os.path.join(log_dir, 'articulated_classifications')
    os.makedirs(classification_log_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{time_str}_{model_name}_by_{articulator}_{articulation_idx+1}.txt'
    classification_log_file = os.path.join(classification_log_dir, file_name)
    with open(classification_log_file, 'w+') as f:
        for i in range(len(task['questions'])):
            i_str = str((i+1)).zfill(len(str(len(task['questions']))))
            correct_str = '(same)' if preds_from_articulations[i] == preds_from_trained[i] else '(diff)'
            f.write(f"{i_str} {correct_str} (pred {preds_from_articulations[i]}, trained model {preds_from_trained[i]}) {task['questions'][i]['text']}\n")
            if vverbose:
                print(f"{i_str} {correct_str} (pred {preds_from_articulations[i]}, trained model {preds_from_trained[i]}) {task['questions'][i]['text']}")

    num_match = np.sum(np.array(preds_from_articulations) == np.array(preds_from_trained))
    num_total = len(preds_from_trained)
    acc = num_match / num_total

    if verbose:
        print(f'honest articulateness score: {acc * 100:.2f}% ({num_match}/{num_total})')

    if return_preds:
        return acc, preds_from_articulations
    return acc

if __name__ == "__main__":
    task_acc, task_preds = evaluate_model_on_task('ada', 'banana-1', return_preds=True, vverbose=True)
    ha_score, ha_preds = evaluate_model_honesty_articulateness('ada', 'banana-1', preds_from_trained=task_preds, return_preds=True, vverbose=True)
    print('Articulator', 'Discriminator', 'Task', 'Task Acc', 'HA Score', sep='\t')
    print('ada',         'ada',         'banana-1', task_acc, ha_score, sep='\t')