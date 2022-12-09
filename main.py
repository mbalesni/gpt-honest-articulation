import numpy as np

from src.openai_model import OpenAIGPT3
from src.json_task import load_json_task

models = [
    'ada',
    'babbage',
    'curie',
    'davinci',
    'text-davinci-003',
    'code-davinci-002',
    # fine-tuned davinci
]

def evaluate_classification_accuracy(model, task_name, few_shot=True):
    task = load_json_task(task_name)

    few_shot_examples = [f'Example: "{example["text"]}" is class {example["label"]}' for example in task['few_shot_examples']] if few_shot else []
    few_shot_examples_str = '\n'.join(few_shot_examples)
    test_examples = [f'Example: "{example["text"]}" is class' for example in task['test_examples']]
    test_examples_labels = [example['label'] for example in task['test_examples']]

    common_prompt = task['instruction'] + '\n' + few_shot_examples_str + '\n'
    prompts = [common_prompt + test_example for test_example in test_examples]

    outputs = model.generate_text(prompts, max_length=task['max_length'], stop_string=task['stop_string'])

    preds = []
    for output in outputs:
        try:
            output = output.strip()
            output = int(output)
        except:
            output = -1
        preds.append(output)
    
    print('preds:')
    print(preds)
    print('true:')
    print(test_examples_labels)

    num_correct = np.sum(np.array(preds) == np.array(test_examples_labels))
    num_total = len(test_examples_labels)
    acc = num_correct / num_total

    print(f'accuracy: {acc * 100:.2f}% ({num_correct}/{num_total})')
    return acc


if __name__ == "__main__":
    model = OpenAIGPT3('code-davinci-002')
    
    evaluate_classification_accuracy(model, 'banana')
