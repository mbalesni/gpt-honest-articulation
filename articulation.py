from src.openai_model import OpenAIGPT3
from src.json_task import load_json_task, make_few_shots

def add_prefixes(explanations, prefixes):
    return [prefix + explanation for prefix, explanation in zip(prefixes, explanations)]

def articulate(model_name, task_name=None, task_prompt=None, explanation_prompt=None, few_shot=True, max_length=200, stop_string=None, log_dir='logs'):
    model = OpenAIGPT3(model_name, log_dir=log_dir)

    if task_name is None:
        assert task_prompt is not None, 'Must provide either task_name or task_prompt and explanation_prompt'
        assert explanation_prompt is not None, 'Must provide either task_name or task_prompt and explanation_prompt'

    if task_name is not None:
        task = load_json_task(task_name)
        if task_prompt is None:
            task_prompt = task['instruction'] + (make_few_shots(task) if few_shot else '')
        explanation_prompt = explanation_prompt or task['explanation_prompts']

    if isinstance(explanation_prompt, list):
        articulation_prompt = [task_prompt.strip() + '\n\n' + exp_prompt for exp_prompt in explanation_prompt]
    else:
        articulation_prompt = task_prompt.strip() + '\n\n' + explanation_prompt
    
    if stop_string is None and task_name is not None:
        stop_string = task['explanation_stop_string']

    explanations = model.generate_text(articulation_prompt, max_length=max_length, stop_string=stop_string)

    if not isinstance(explanations, list):
        explanations = [explanations]

    explanations = add_prefixes(explanations, explanation_prompt)
    
    return explanations
