# Policy Training
## Quickstart
You can see the behavior of pre-trained policy with
```bash
PYTORCH_JIT=0 python3 show_ppo_arm.py +platform=debug +env=icra_base +run=icra_ours ++env.seed=56081 ++eval_period=-1 ++tag=policy ++global_device=cuda:0 ++path.root=/tmp/pkm/ppo-a ++icp_obs.icp.ckpt=imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920 ++load_ckpt=imm-unicorn/corn-public:dr-icra_base-icra_ours-ours-final-000042 ++env.num_env=16
```
**Note**
You may have to setup display following [Instruction](#extra-tips) for visualization.

## Policy Training

For policy training run the training script as follows:

```bash
PYTORCH_JIT=0 python3 train_ppo_arm.py +platform=debug +env=icra_base +run=icra_ours ++env.seed=56081 ++eval_period=-1 ++tag=policy ++global_device=cuda:0 ++path.root=/tmp/pkm/ppo-a ++icp_obs.icp.ckpt="${CORN_CKPT}"
```

Replace `${CORN_CKPT}` with the name of the pretrained model that you have trained or downloaded.

Alternatively, you may use `CORN_CKPT=imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920` to use our pretrained weights.

If the training run results in an error, you may need to run:
```bash
sudo chmod -R 777 /home/user/.cache/
```

By default, the results of the training will be stored in `/tmp/pkm/ppo-arm/run-{:03d}`.

In general, we disable JIT for the purposes of the current code release, as its stability depends on your hardware.
In the case that your particular GPU+docker setup supports JIT compilation, you may enable `torch.jit.script()` as follows:
```bash
PYTORCH_JIT=1 python3 ...
```


## Policy Evaluation

```bash
PYTORCH_JIT=0 python3 show_ppo_arm.py +platform=debug +env=icra_base +run=icra_ours ++env.seed=56081 ++eval_period=-1 ++tag=policy ++global_device=cuda:0 ++path.root=/tmp/pkm/ppo-a ++icp_obs.icp.ckpt="${CORN_CKPT}" ++load_ckpt="${POLICY_CKPT}" ++env.num_env=1
```
Replace `${POLICY_CKPT}` with the name of the policy that you have trained or downloaded.

Alternatively, you may use `POLICY_CKPT=imm-unicorn/corn-public:dr-icra_base-icra_ours-ours-final-000042` to use our pretrained weights.

By default, we disable JIT for the purposes of the current code release.
In the case that your particular GPU+docker setup supports JIT compilation, you may enable `torch.jit.script()` as follows:
```bash
PYTORCH_JIT=1 python3 ...
```

Assuming all goes well, you should see an image like this:

![policy-image](../../../fig/policy.png)

To enable graphics rendering, see [extra tips](#extra-tips).


## Sim2real Teacher-Student Distillation

To distill the privileged teacher to the student, first run the second phase of policy fine-tuning to reduce the action-space:

```bash
PYTORCH_JIT=0 python3 train_ppo_arm.py +platform=debug +env=icra_base +run=icra_ours ++env.seed=56081 ++eval_period=-1 ++tag=student ++global_device=cuda:0 ++path.root=/tmp/pkm/ppo-a ++env.num_env=8192 ++is_phase2=true ++phase2.min_reset_to_update=65536 ++agent.train.lr=2e-6 ++agent.train.alr.initial_scale=6.67e-3 ++icp_obs.icp.ckpt="${CORN_CKPT}"  ++load_ckpt="${POLICY_CKPT}"
```

Afterward, run the distillation script as follows:

```bash
PYTORCH_JIT=0 python3 train_rma.py +platform=debug +env=icra_base_rma_mc +run=icra_ours +student=rma_gru_student_base_v2 ++env.seed=56081 ++env.num_env=2048 ++eval_period=-1 ++tag=dagger ++global_device=cuda:0 ++path.root=/tmp/pkm/rma ++icp_obs.icp.ckpt="${CORN_CKPT}"  ++load_ckpt="${POLICY_CKPT}" ++train_student_policy=0 ++dagger=true ++is_phase2=true ++dagger_train_env.deterministic_action=false ++phase2.start_dof_pos_offset=0.03 ++phase2.adaptive_residual_scale=false ++env.franka.max_pos=0.06 ++env.franka.max_ori=0.1
```

As before, replace `${POLICY_CKPT}` and `${CORN_CKPT}` as necessary.

## Extra Tips

To enable the visualization during [policy training](#policy-training) or [policy evaluation](#policy-evaluation), you may need to add `++env.use_viewer=1` to the command line arguments.

Note that this will slow down the training process, so it's generally not recommended.

In case the visualization window does not start, ensure that the `${DISPLAY}` environment variable is configured to match that of the host system, which can be checked by running:
```bash
echo $DISPLAY
```
_outside_ of the docker container, i.e. in your _host_ system. Then, _inside_ the docker container:

```bash
export $DISPLAY=...
```
so that it matches the output of the earlier command (`$echo ${DISPLAY}`) in your host system.

## Experiment Tracking & API Setup

While the models and training progressions are also stored locally, You may track our model progress via [WanDB](https://wandb.ai/) and store the pretrained models with [HuggingFace](https://huggingface.co/).
To configure both APIs, run `wandb login` / `huggingface-cli login`.

Afterward, you can replace the `+platform=debug` directive in the command line with `+platform=dr`.

## Troubleshooting

In the case that a prior `jit` compilation has failed, it may be necessary to clear the jit cache:

```
rm -rf ~/.cache/torch_extensions
```
