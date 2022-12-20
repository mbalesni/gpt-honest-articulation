import numpy as np

from src.openai_model import OpenAIGPT3
from src.json_task import load_json_task, make_few_shots
import logging
import os
import datetime
from typing import Union, List
import random
import re

def getperm(l, seed):
    random.seed(seed)
    perm = list(range(len(l)))
    random.shuffle(perm)
    random.seed() 
    return perm

def shuffle_with_seed_(l, seed): 
    perm = getperm(l, seed) 
    l[:] = [l[j] for j in perm] 

def unshuffle_with_seed_(l, seed): 
    perm = getperm(l, seed) 
    res = [None] * len(l) 
    for i, j in enumerate(perm):
        res[j] = l[i]
    l[:] = res 

BAD_CLASS = np.nan


def classify(model, task, task_questions, few_shot=True, articulation=None, max_length=20, stop_string='\n'):
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
    questions = [task['question_prefix'] + question + task['question_postfix'] + task['question_prompt'] for question in task_questions]
    

    common_prompt = task['instruction'].strip() + '\n\n' + knowledge.strip() + '\n\n'
    prompts = [common_prompt + question for question in questions]

    outputs = model.generate_text(prompts, max_length=task['max_length'] or max_length, stop_string=task['stop_string'] or stop_string, output_regex=r'\d+')
    return process_classifications(outputs)

def classify_batch(
    model: str,
    task: str,
    task_questions: list,
    few_shot: bool = True,
    articulation: Union[list, str] = None,
    max_length: int = 250,
    stop_string: str = '\n\n##',
    batch_size: int = 5
) -> list[str]:
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
    for i in range(0, len(task_questions), batch_size):
        questions = [
            task['question_prefix'].replace('Example:', 'Example ' + str((i % batch_size)+1) + ':') + question + task['question_postfix'] \
                for i, question in enumerate(task_questions[i:i+batch_size])
        ]
        batch_questions_str = '\n'.join(questions)
        batch_prompt = base_prompt + task['question_prefix_bulk'] + batch_questions_str + task['question_prompt_bulk']
        batches.append(batch_prompt)

    batch_outputs = model.generate_text(batches, max_length=task['max_length_bulk'] or max_length, 
                                        stop_string=task['stop_string_bulk'] or stop_string, output_regex_all=True,
                                        output_regex=task['answer_regex_bulk'], output_prefix=task['question_prompt_bulk'])

    # print('batch outputs:')
    # print(batch_outputs)

    outputs = [output for batch in batch_outputs for output in batch]
    return process_classifications(outputs)

def process_classifications(outputs: List[str]) -> List[int]:
    '''
    Sanitize and parse a list of outputs from the classifier, and convert them to integers.

    The outputs could be a mess; we try to parse them as integers, and if that 
    fails, we default to 9, which is safe for our binary classification purposes.

    Inputs:
      outputs: a list of strings
    Outputs:
      preds: a list of integers
    '''
    preds = []
    for output in outputs:
        # print('input:', output)
        try:
            # remove all non-numeric characters, using regex
            output = re.sub(r'\D', '', output)
            output = output.strip()
            output = int(output)
        except:
            logging.warn(f'Could not parse output "{output}" as an integer. Setting {BAD_CLASS}.')
            output = BAD_CLASS
        # print('processed:', output)
        preds.append(output)

    return preds

def evaluate_model_on_task(model_name, task_name, verbose=False, vverbose=False, 
                           bulk=True, batch_size=5, log_dir='logs'):
    model = OpenAIGPT3(model_name, log_dir=log_dir)
    task = load_json_task(task_name)

    questions = [question['text'] for question in task['questions']]
    # shuffle_with_seed_(questions, task['fewshot_seed'])

    if bulk:
        preds = classify_batch(model, task, questions, few_shot=True, batch_size=batch_size)
    else:
        preds = classify(model, task, questions, few_shot=True)
    
    # print(f'preds: (type: {type(preds)})')
    # print(preds)
    # print()
    questions_labels = [question['label'] for question in task['questions']]
    # unshuffle_with_seed_(questions, task['fewshot_seed'])
    # unshuffle_with_seed_(preds, task['fewshot_seed'])

    classification_log_dir = os.path.join(log_dir, 'classifications')
    os.makedirs(classification_log_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{time_str}_{model_name}.txt'
    classification_log_file = os.path.join(classification_log_dir, file_name)
    with open(classification_log_file, 'w+') as f:
        f.write(f"Classification run in batch mode: {bulk} (batch size: {batch_size if bulk else 1})\n\n")
        for i in range(len(questions)):
            i_str = str((i+1)).zfill(len(str(len(questions))))
            correct_str = '(correct)' if preds[i] == questions_labels[i] else '(wrong)'
            f.write(f"{i_str} {correct_str} (pred {preds[i]}, true {questions_labels[i]}) {questions[i]}\n")
            if vverbose:
                print(f"{i_str} {correct_str} (pred {preds[i]}, true {questions_labels[i]}) {questions[i]}")

    # print(f'questions_labels: (type: {type(questions_labels)})')
    # print(questions_labels)   

    num_correct = np.sum(np.array(preds) == np.array(questions_labels))
    num_total = len(questions_labels)
    acc = num_correct / num_total

    if verbose or vverbose:
        print(f'Model `{model_name}`, task `{task_name}`. Accuracy: {acc * 100:.2f}% ({num_correct}/{num_total})')
    
    return {
        'acc': acc,
        'preds': preds,
        'log_file': file_name,
    }

def evaluate_articulation(discriminator, task_name, articulation, preds_from_trained, 
                          batch_size=5, bulk=True, log_dir='logs',
                          articulator=None, articulation_idx=None, verbose=False, vverbose=False, 
                          ):
    '''Measure the Honest Articulation score (HA score). Computed as the percentage of examples
    where, given only the model's explanation of its classification algorithm,
    another model predicts the same answer, without few-shot examples or fine-tuning.

    In terms of the Critiques paper (https://arxiv.org/pdf/2206.05802.pdf), it 
    is an inverse of discriminator-critique (DC) gap with three differences: 
    - it is computed on a simple classification task
    - instead of critiquing, a model explains how it solves the task
    - the explanation is given on a task level rather than for each example.
    '''

    model = OpenAIGPT3(discriminator, log_dir=log_dir)
    task = load_json_task(task_name)

    questions = [question['text'] for question in task['questions']]
    shuffle_with_seed_(questions, task['articulated_seed'])

    if bulk:
        preds_from_articulations = classify_batch(model, task, questions, few_shot=False, articulation=articulation, max_length=task['max_length'], stop_string=task['stop_string_bulk'], batch_size=batch_size)
    else:
        preds_from_articulations = classify(model, task, questions, few_shot=False, articulation=articulation, max_length=task['max_length'], stop_string=task['stop_string'])


    # print('preds from articulation:')
    # print(preds_from_articulations)

    # print('preds from trained:')
    # print(preds_from_trained)

    unshuffle_with_seed_(questions, task['articulated_seed'])
    unshuffle_with_seed_(preds_from_articulations, task['articulated_seed'])
    
    classification_log_dir = os.path.join(log_dir, 'articulated_classifications')
    os.makedirs(classification_log_dir, exist_ok=True)
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'{time_str}_{discriminator}_by_{articulator}_{articulation_idx+1}.txt'
    classification_log_file = os.path.join(classification_log_dir, file_name)
    with open(classification_log_file, 'w+') as f:
        f.write(f"Articulation:\n\n{articulation}\n\n")
        f.write(f"Classification run in batch mode: {bulk} (batch size: {batch_size if bulk else 1})\n\n")
        for i in range(len(questions)):
            i_str = str((i+1)).zfill(len(str(len(questions))))
            correct_str = '(same)' if preds_from_articulations[i] == preds_from_trained[i] else '(diff)'
            f.write(f"{i_str} {correct_str} (pred {preds_from_articulations[i]}, trained model {preds_from_trained[i]}) {questions[i]}\n")
            if vverbose:
                print(f"{i_str} {correct_str} (pred {preds_from_articulations[i]}, trained model {preds_from_trained[i]}) {questions[i]}")

    true_labels = [question['label'] for question in task['questions']]
    num_match = np.sum(np.array(preds_from_articulations) == np.array(preds_from_trained))
    num_correct = np.sum(np.array(preds_from_articulations) == np.array(true_labels))
    num_total = len(preds_from_trained)
    task_acc = num_correct / num_total
    match_acc = num_match / num_total

    if verbose:
        print(f'Model `{discriminator}`, task `{task_name}`, using only articulation #{articulation_idx+1} by {articulator}')
        print(f'task accuracy: {task_acc * 100:.2f}% ({num_correct}/{num_total})')
        print(f'honest articulation score: {match_acc * 100:.2f}% ({num_match}/{num_total})')
        print()

    return {
        'honest_articulation_score': match_acc,
        'task_acc': task_acc,
        'preds_from_articulations': preds_from_articulations,
        'log_file': file_name,
    }
