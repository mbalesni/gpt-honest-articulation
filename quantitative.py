import pandas as pd
import time
import os
import argparse

from articulation import articulate
from evaluation import evaluate_model_on_task, evaluate_articulation
from src.json_task import load_json_task

RESULTS_BASEDIR = 'results'

def run_experiments(task_name, articulators, discriminators, continue_from=None):
    """
    Run experiments for a given task, list of articulators, and list of discriminators.

    If continue_from is specified, will continue from the specified csv file.
    """

    save_path = continue_from or os.path.join(RESULTS_BASEDIR, f'{time_str}_fewshot_{task_name}.csv') 

    if continue_from:
        print('Continuing experiment from', continue_from)
        df_results = pd.read_csv(continue_from)
    else:
        print('Starting new experiment')
        df_results = pd.DataFrame(columns=['articulator', 'task_name', 'acc_fewshot', 'discriminator', 'explanation_idx', 'acc_articulated', 'honest_articulation_score'])
    
    print(df_results)
    print()

    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f'{RESULTS_BASEDIR}/{time_str}/{task_name}'
    os.makedirs(results_dir, exist_ok=True)

    articulation_stop_strings = {
        'code-cushman-001': '\n\n#',
        'code-davinci-002': '\n\n#',
        'text-ada-001': None,
        'text-babbage-001': None,
        'text-curie-001': None,
        'text-davinci-002': None,
        'text-davinci-003': None,
    }

    num_explanations = len(load_json_task(task_name)['explanation_prompts'])

    try:

        for articulator in articulators:

            if 'ada' in articulator:
                bulk = False # ada too dumb to follow batch request pattern
            else:
                bulk = True

            # if results already exist, skip
            results_for_this_articulator = df_results[(df_results['articulator'] == articulator) & (df_results['task_name'] == task_name)]
            if results_for_this_articulator.shape[0] == num_explanations * len(discriminators):
                continue

            task_acc_fewshot, preds_fewshot = evaluate_model_on_task(articulator, task_name, return_preds=True, 
                                                                    verbose=True, vverbose=False, log_dir=results_dir,
                                                                    bulk=bulk)
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

                    if 'code' in discriminator: 
                        time.sleep(6)

                    honest_articulation_score, task_acc_articulated = evaluate_articulation(discriminator, task_name, explanation, preds_from_trained=preds_fewshot, 
                                                                                            verbose=True, log_dir=results_dir, articulation_idx=i, articulator=articulator)
                    result_row = {'articulator': articulator, 'discriminator': discriminator, 'task_name': task_name, 'acc_fewshot': task_acc_fewshot, 'explanation_idx': i, 'acc_articulated': task_acc_articulated, 'honest_articulation_score': honest_articulation_score}
                    df_results = pd.concat([df_results, pd.DataFrame([result_row])], ignore_index=True)

        # verify the number of results is correct
        assert df_results.shape[0] == num_explanations * len(discriminators) * len(articulators), f'Expected {num_explanations * len(discriminators) * len(articulators)} results, got {df_results.shape[0]}'
        df_results.to_csv(save_path, index=False)
        print('Saved results to', save_path)

    except Exception as e:
        print(e)
        df_results.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '--task-name', type=str, default='banana-1')
    parser.add_argument('--articulators', type=str, nargs='+', default=['text-ada-001', 'text-babbage-001', 'code-cushman-001', 'text-curie-001', 'text-davinci-002', 'text-davinci-003', 'code-davinci-002'])
    parser.add_argument('--discriminators', type=str, nargs='+', default=['text-davinci-003', 'code-davinci-002'])
    parser.add_argument('--continue_from', type=str, default=None, help='path to results csv file to continue from')
    args = parser.parse_args()

    run_experiments(args.task, args.articulators, args.discriminators, args.continue_from)
