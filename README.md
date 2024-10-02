# Hybrid Spatial Representations for Species Distribution Modeling

We use the same version of iNaturalist data as in SINR, which can be obtained as follows:

```bash
curl -L https://data.caltech.edu/records/b0wyb-tat89/files/data.zip --output data.zip
unzip -q data.zip
rm data.zip
```

Afterwards, install requirements with:

```bash
conda create --name hsrsdm python=3.10
conda activate hsrsdm
pip install -r requirements.txt
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Then run `python train_and_evaluate_models.py` to get results. Key hyper-parameters (learning rate, implicitness, and observation cap) can be directly adjusted in the file.

## Model Zoo
The model zoo can be found at Open Science Framework: [https://osf.io/pbk9a/?view_only=e4b3ced52eeb460a9f8cd8dd1569a0df](https://osf.io/pbk9a/?view_only=e4b3ced52eeb460a9f8cd8dd1569a0df)

## Acknowledgements

This codebase borrows from [SINR](https://github.com/elijahcole/sinr), [NeRFStudio](https://github.com/nerfstudio-project/nerfstudio), and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). We would like to thank the authors of those works.
