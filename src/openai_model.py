import os
import scipy
import numpy as np
import tempfile
import dotenv 
dotenv.load_dotenv()

import datetime

import transformers
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

import openai
from src import model_utils

openai.api_key = os.getenv("OPENAI_API_KEY", None)
# if no key in env, ask user (with a secure prompt):
if openai.api_key is None:
    import getpass
    openai.api_key = getpass.getpass("Please paste your OpenAI API key: ")


def with_logger(func, dir):

    def complete_with_logging(*args, **kwargs):
        batch_outputs = func(*args, **kwargs)
        prompt = kwargs['prompt']
        time_str = datetime.datetime.fromtimestamp(batch_outputs.created).strftime('%Y-%m-%d_%H:%M:%S')
        file_base_name = f'{time_str}_{batch_outputs.model}_{batch_outputs.usage.prompt_tokens}_{batch_outputs.usage.completion_tokens}'
        
        for i, completion in enumerate(batch_outputs.choices):
            file_name = f'{file_base_name}-{i+1}of{len(batch_outputs.choices)}.md'
            with open(os.path.join(dir, file_name), 'w+') as logfile:
                logfile.write(prompt[i])
                logfile.write('**' + completion.text + '**')
        return batch_outputs

    return complete_with_logging

class OpenAIGPT3:
    def __init__(self, model='ada', max_parallel=20, log_dir=None):
        self.queries = []
        self.model = model
        self.max_parallel = max_parallel
        if log_dir:
            log_dir = os.path.join(log_dir, 'completions')
        self.log_dir = log_dir or tempfile.mkdtemp()
        os.makedirs(self.log_dir, exist_ok=True)

    def generate_text(
        self, inputs, max_length=500, stop_string=None, output_regex=None,
        temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0,
        output_prefix=None, output_postfix=None, output_regex_all=False
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = []

        n_batches = int(np.ceil(len(inputs) / self.max_parallel))
        for batch_idx in range(n_batches):
            batch_inputs = inputs[
                batch_idx * self.max_parallel : (batch_idx + 1) * self.max_parallel
            ]
            batch_outputs = with_logger(openai.Completion.create, self.log_dir)(
                engine=self.model,
                frequency_penalty=frequency_penalty,
                max_tokens=max_length,
                presence_penalty=presence_penalty,
                prompt=batch_inputs,
                stop=stop_string,
                stream=False,
                temperature=temperature,
                top_p=top_p,
            )
            for completion in batch_outputs.choices:
                outputs.append(completion.text)

        # add prefixes and postfixes
        outputs = [ (output_prefix or '') + output + (output_postfix or '') for output in outputs]
        
        if len(inputs) == 1:
            outputs = outputs[0]
        
        outputs = model_utils.postprocess_output(
            outputs, stop_string, output_regex, output_regex_all
        )
        return outputs

    def flatten_multiple_choice_examples(self, inputs, targets):
        flat_idx = []
        flat_inputs = []
        flat_choices = []
        for example_id, (example_input, choices) in enumerate(zip(inputs, targets)):
            for choice_id, choice in enumerate(choices):
                flat_idx.append((example_id, choice_id))
                flat_inputs.append(example_input)
                flat_choices.append(choice)

        return flat_idx, flat_inputs, flat_choices

    def get_target_logprobs(self, completion, target):
        '''Naive implementation of getting the logprobs of the target:
        
        To find out which tokens the target is made of, the function iteratively 
        concatenates returned tokens from the end, and compares a running 
        concatenation with the target.
        '''
        cum_sum = ''
        for i, token in enumerate(reversed(completion.logprobs['tokens'])):
            cum_sum = token + cum_sum
            if cum_sum.strip() == target.strip():
                break

        target_tokens_logprobs = completion.logprobs['token_logprobs'][-(i+1):]
        if None in target_tokens_logprobs:
            print('Found None in target_tokens_logprobs:', target_tokens_logprobs, 'in completion:', completion)
        return sum(target_tokens_logprobs)

    def cond_log_prob(self, inputs, targets, absolute_normalization=False):

        if isinstance(targets, str):
            targets = [targets]

        if isinstance(inputs, str):
            inputs = [inputs]
            targets = [targets]

        flat_idx, flat_inputs, flat_choices = self.flatten_multiple_choice_examples(
            inputs=inputs, targets=targets
        )
        num_examples = len(flat_idx)
        flat_scores = []
        batch_size = self.max_parallel
        for idx in range(0, num_examples, batch_size):
            batch_idx = flat_idx[idx : min(idx + batch_size, num_examples)]
            batch_inputs = flat_inputs[idx : min(idx + batch_size, num_examples)]
            batch_choices = flat_choices[idx : min(idx + batch_size, num_examples)]

            batch_queries = [inpt + target for inpt, target in zip(batch_inputs, batch_choices)]
            batch_outputs = openai.Completion.create(
                model=self.model,
                prompt=batch_queries,
                max_tokens=0,
                temperature=0,
                logprobs=1,
                echo=True,
            )

            for i, completion in enumerate(batch_outputs.choices):
                target_logprobs = self.get_target_logprobs(completion, batch_choices[i])
                flat_scores.append(target_logprobs)

        scores = [[] for _ in range(len(inputs))]

        for idx, score in zip(flat_idx, flat_scores):
            if score == 0:
              # all tokens were masked. Setting score to -inf.
              print('Found score identical to zero. Probably from empty target. '
                             'Setting score to -inf.'
                            )
              scores[idx[0]].append(-np.inf)
            else:
              scores[idx[0]].append(score)

        if not absolute_normalization:
            scores = [
                list(score_row - scipy.special.logsumexp(score_row))
                for score_row in scores
            ]

        if len(inputs) == 1:
            scores = scores[0]

        return scores