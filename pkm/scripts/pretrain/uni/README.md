# Pretraining

## Run Pretraining Script

After setting up the dataset in e.g. `/tmp/col-12-2048`, run the training script:

```bash
python3 train_col.py
```

If the training run results in an error, you may need to run:
```bash
sudo chmod -R 777 /home/user/.cache/
```

By default, the results of the pretraining will be stored in `/tmp/corn-/col/run-{:03d}`.

## Experiment Tracking API Setup

While the models and training progressions are also stored locally,
we track our model progress via [WanDB](https://wandb.ai/) and store the pretrained models with [HuggingFace](https://huggingface.co/).
To configure both APIs, run `wandb login` / `huggingface-cli login`.

By default, `use_wandb` and `use_hfhub` arguments are disabled inside the script, which means that the remote tracking pipeline
is disabled. If you would like to enable this, then set the corresponding arguments to `true` within the file.
