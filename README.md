# (ICLR 2024) CORN: Contact-based Object Representation for Nonprehensile Manipulation of General Unseen Objects

[Yoonyoung Cho](yycho0108.github.io/research) [Junhyek Han](https://junhyekh.github.io/) [Yoontae Cho](imsquared.github.io) [Beomjoon Kim](beomjoonkim.github.io) | [Intelligent Mobile Manipulation (IM^2) Lab](imsquared.github.io)

Korea Advanced Institute of Science and Technology (KAIST) Kim Jaechul Graduate school of AI

International Conference on Learning Representations (ICLR) 2024 ([OpenReview](https://openreview.net/forum?id=KTtEICH4TO))

![https://www.youtube.com/watch?v=TQE-Wku_2sk](fig/thumb-compressed.gif)

* [website](https://sites.google.com/view/contact-non-prehensile)
* [video](https://www.youtube.com/watch?v=TQE-Wku_2sk)

# Table of Contents

- 1 [Workspace Setup](#setup)
  - 1.1 [Docker setup](#docker-setup)
  - 1.2 [Package Setup](#package-setup)
  - 1.3 [Assets Setup](#assets-setup)
- 2 [Pretraining](#pretraining)
- 3 [Policy Training](#policy-training)
- 4 [Real-World Experiments](#real-world-experiments)
  - 4.1 [Sim2Real Distillation](#sim2real-distillation)
  - 4.2 [Real-World Deployment](#real-world-deployment)

# Setup

## Docker Setup

Refer to the instructions in [docker](./docker)

## Package Setup

### Isaac Gym

First, download isaac gym from [here](https://developer.nvidia.com/isaac-gym) and extract them to the `${IG_PATH}` host directory
that you configured during docker setup. By default, we assume this is `/home/corn/isaacgym`, which maps to`/opt/isaacgym` directory inside the container.
In other words, the resulting directory structure should look like:

```bash
$ tree /opt/isaacgym -d -L 1

/opt/isaacgym
|-- assets
|-- docker
|-- docs
|-- licenses
`-- python
```

(If `tree` command is not found, you may simply install it via `sudo apt-get install tree`.)

Afterward, follow the instructions in the referenced page to install the isaac gym package.

Alternatively, assuming that the isaac gym package has been downloaded and extracted in the correct directory(`/opt/isaacgym`),
we provide the default setups for isaac gym installation in the [setup script](./setup.sh)
in the [following section](#python-package), which handles the installation automatically.

### Python Package

Then, inside the docker image (assuming `${PWD}` is the repo root), run the [setup script](./setup.sh):

```bash
bash setup.sh
```


To test if the installation succeeded, you can run:
```bash
python3 -c 'import isaacgym; print("OK")'
python3 -c 'import pkm; print("OK")'
```

## Assets Setup

We release a pre-processed version of the object mesh assets from [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) in [here](https://huggingface.co/imm-unicorn/corn-public/resolve/main/DGN.tar.gz).

After downloading the assets, extract them to `/path/to/data/DGN` in the _host_ container, so that `/path/to/data` matches the directory
configured in [docker/run.sh](docker/run.sh), i.e.

```bash
mkdir -p /path/to/data/DGN
tar -xzf DGN.tar.gz -C /path/to/data/DGN
```

so that the resulting directory structure _inside_ the docker container looks as follows:

```bash
$ tree /input/DGN --filelimit 16 -d     

/input/DGN
|-- coacd
`-- meta-v8
    |-- cloud
    |-- cloud-2048
    |-- code
    |-- hull
    |-- meta
    |-- new_pose
    |-- normal
    |-- normal-2048
    |-- pose
    |-- unique_dgn_poses
    `-- urdf
```

# Pretraining

Navigate to the pretraining directory in `pkm/scripts/pretrain` and follow the instructions in the [README](./pkm/scripts/pretrain/README.md).

# Policy Training

Navigate to the policy training directory in `pkm/scripts/train` and follow the instructions in the [README](./pkm/scripts/train/README.md).

# Real World Experiments

## Sim2Real Distillation

Before running the real-world experiments, we distill the privileged teacher into a student that can operate with observations available in the real world.

See the related scripts and documentation in the [README](./pkm/scripts/train/README.md#sim2real-teacher-student-distillation).

## Real World Deployment

Navigate to the real-world deployment directory in `pkm/scripts/real` and follow the instructions in the [README](./pkm/scripts/real/README.md).

# Citation

If you find this codebase useful, consider citing:

```
@inproceedings{cho2024corn,
  title={{CORN: Contact-based Object Representation for Nonprehensile Manipulation of General Unseen Objects}},
  author={Cho, Yoonyoung and Han, Junhyek and Cho, Yoontae and Kim, Beomjoon},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```
