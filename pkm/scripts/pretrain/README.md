# Pretraining

## Data Generation

To generate pretraining data, see inside [gen_col](./gen_col) directory.

Alternatively, we provide pre-generated data [here](https://huggingface.co/imm-unicorn/corn-public/blob/main/col-12-2048.tar.gz).

To download and use the data, run the following inside the docker container:

```bash
cd /tmp/
wget https://huggingface.co/imm-unicorn/corn-public/resolve/main/col-12-2048.tar.gz
tar -xzf col-12-2048.tar.gz
```

## Training

To pretrain the representation model, follow the instructions inside the [uni](./uni) directory.

Alternatively, we provide pretrained checkpoints [here](https://huggingface.co/imm-unicorn/corn-public/blob/main/512-32-balanced-SAM-wd-5e-05-920).
