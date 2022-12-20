import pandas as pd
import time
import os
from collections import defaultdict
import argparse
import logging
import traceback

from src.evaluation import evaluate_model_on_task, evaluate_articulation, articulate
from src.json_task import load_json_task

logging.basicConfig(level=logging.WARNING)

RESULTS_BASEDIR = 'results'

MODELS_THAT_CANNOT_BULK = ['ada', 'text-ada-001']

def run_experiments(task_name, articulators, discriminators, continue_from=None, bulk=True):
    """
    Run experiments for a given task, list of articulators, and list of discriminators.

    If continue_from is specified, will continue from the specified csv file.
    """

    if continue_from:
        print('Continuing experiment from', continue_from)
        df_results = pd.read_csv(continue_from)
    else:
        print('Starting new experiment')
        df_results = pd.DataFrame(columns=[
            'articulator', 
            'task_name', 
            'acc_fewshot', 
            'discriminator', 
            'explanation_idx', 
            'acc_articulated', 
            'honest_articulation_score',
            'path_to_classification_log',
            'path_to_articulated_classification_log',
        ])
    
    print(df_results)
    print()

    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f'{RESULTS_BASEDIR}/{time_str}/{task_name}'
    os.makedirs(results_dir, exist_ok=True)
    results_csv_path = continue_from or os.path.join(RESULTS_BASEDIR, f'{time_str}_fewshot_{task_name}.csv') 

    articulation_stop_strings = defaultdict(lambda: None, {
        'code-cushman-001': '\n\n#',
        'code-davinci-002': '\n\n#',
        'text-ada-001': None,
        'text-curie-001': None,
        'text-davinci-001': None,
        'text-davinci-002': None,
        'text-davinci-003': None,
    })

    num_explanations = len(load_json_task(task_name)['explanation_prompts'])

    initial_n_results = df_results.shape[0]

    try:

        for articulator in articulators:

            articulator_bulk = bulk if articulator not in MODELS_THAT_CANNOT_BULK else False

            # if results already exist, skip
            results_for_this_articulator = df_results[(df_results['articulator'] == articulator) & (df_results['task_name'] == task_name)]
            if results_for_this_articulator.shape[0] == num_explanations * len(discriminators):
                continue

            task_results = evaluate_model_on_task(articulator, task_name, verbose=True, vverbose=False, 
                                                  log_dir=results_dir, bulk=articulator_bulk)
            task_acc_fewshot = task_results['acc'] 
            preds_fewshot = task_results['preds']
            path_to_classification_log = task_results['log_file']

            explanations = articulate(articulator, task_name, log_dir=results_dir,
                                    stop_string=articulation_stop_strings[articulator])

            for discriminator in discriminators:

                # if results already exist, skip
                results_for_this_discriminator = df_results[(df_results['articulator'] == articulator) & (df_results['discriminator'] == discriminator) & (df_results['task_name'] == task_name)]
                if results_for_this_discriminator.shape[0] == num_explanations:
                    continue

                for i, explanation in enumerate(explanations):

                    # if results already exist, skip
                    results_for_this_explanation = df_results[(df_results['articulator'] == articulator) & (df_results['discriminator'] == discriminator) & (df_results['task_name'] == task_name) & (df_results['explanation_idx'] == i)]
                    if results_for_this_explanation.shape[0] == 1:
                        continue

                    articulation_results = evaluate_articulation(discriminator, task_name, explanation, preds_from_trained=preds_fewshot, 
                                                                verbose=True, log_dir=results_dir, articulation_idx=i, articulator=articulator, bulk=bulk)
                    
                    honest_articulation_score = articulation_results['honest_articulation_score']
                    task_acc_articulated = articulation_results['task_acc']
                    path_to_articulated_classification_log = articulation_results['log_file']

                    result_row = {
                        'articulator': articulator, 'discriminator': discriminator, 
                        'task_name': task_name, 'acc_fewshot': task_acc_fewshot, 
                        'explanation_idx': i, 'acc_articulated': task_acc_articulated, 
                        'honest_articulation_score': honest_articulation_score,
                        'path_to_classification_log': path_to_classification_log,
                        'path_to_articulated_classification_log': path_to_articulated_classification_log,
                    }
                    df_results = pd.concat([df_results, pd.DataFrame([result_row])], ignore_index=True)

        # verify the number of results is correct
        n_new_results = df_results.shape[0] - initial_n_results
        assert n_new_results == num_explanations * len(discriminators) * len(articulators), f'Expected {num_explanations * len(discriminators) * len(articulators)} results, got {df_results.shape[0]}'
        df_results.to_csv(results_csv_path, index=False)
        print('Saved results to', results_csv_path)

    except Exception as e:
        print(e)
        # trace 
        traceback.print_exc()
        df_results.to_csv(results_csv_path, index=False)

        print('Saved intermediate results to', results_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '--task-name', type=str, default='banana-1')
    parser.add_argument('--articulators', type=str, nargs='+', default=[
        'ada',
        'babbage',
        'curie',
        'davinci',

        'text-ada-001',
        'text-babbage-001',
        'text-curie-001', 
        'text-davinci-001', 

        'code-cushman-001', 
        'code-davinci-002',

        'text-davinci-002', 
        'text-davinci-003', 
    ])
    parser.add_argument('--discriminators', type=str, nargs='+', default=['code-davinci-002', 'text-davinci-003'])
    parser.add_argument('--no-code-models', action='store_true', help='do not run code models')
    parser.add_argument('--continue_from', type=str, default=None, help='path to results csv file to continue from')
    args = parser.parse_args()

    if args.no_code_models:
        args.articulators = [a for a in args.articulators if 'code' not in a]
        args.discriminators = [d for d in args.discriminators if 'code' not in d]

        assert len(args.articulators) > 0, 'no articulators left, try without --no-code-models'
        assert len(args.discriminators) > 0, 'no discriminators left, try without --no-code-models'

    print('articulators:', args.articulators)
    print('discriminators:', args.discriminators)
    run_experiments(args.task, args.articulators, args.discriminators, args.continue_from, bulk=True)
