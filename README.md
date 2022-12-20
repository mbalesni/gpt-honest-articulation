# Honest articulation of latent knowledge

## Install

1. Install dependencies

```bash
conda create -n halk python=3.10
conda activate halk
pip install ought-ice scipy openai matplotlib seaborn
conda install -n test ipykernel --update-deps --force-reinstall
```

2. Add the OpenAI API key to your environment

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenAI API key. See `.env.example` for inspiration.

## Replicate results

### Compute the results

There're four tasks in this project:
- `banana-1`
- `banana-2`
- `gpt-script-1`
- `gpt-script-2`

For each task, run the following command:
```bash
python run_task.py --task <task-name>
```

If all goes well, when the script finishes you should see a .csv results file appears in the `results` folder. 

If there's an error, intermediate results will be still be saved to the `results` folder, and you can inspect the error message to see what went wrong. Run the script with `--continue_from <path_to_results_csv>` to continue from the last checkpoint. Every task will save into a separate results file, so if your OpenAI account doesn't have a limit on the code models, you can run multiple tasks in parallel.

Running all four tasks will take about 2 hours. If you'd like to speed up the process, you could choose to not evaluate the code models as articulators, and use only the Instruct model for discriminators, like so:

```bash
python run_task.py --task <task-name> --no-code-models
```

This will change the results slightly, but not in a way that affects the conclusions.

### Reproduce the figures

Once all four tasks produced results, run the `results.ipynb` notebook to reproduce the figures in the paper.
