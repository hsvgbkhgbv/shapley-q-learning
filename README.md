# SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning

This is the implementation of the paper SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning (https://arxiv.org/abs/2105.15013).

The implementation is based on PyMARL (https://github.com/oxwhirl/pymarl/). Please refer to that repo for more documentation.

The baselines except for SQDDPG used in this paper are from the repo of Weighted QMIX (https://github.com/oxwhirl/wqmix). To know more about baselines, please refer to that repo.

SQDDPG is implemented based on https://github.com/hsvgbkhgbv/SQDDPG.

The model implemented in this paper is based on Pytorch 1.4.0.

## Included in this repo

In particular implementations for:
- SHAQ

Note that in the repository the naming of certain hyper-parameters and concepts is a little different to the paper:
- $\hat{\alpha}$ in the paper is alpha in the code

## For all SMAC experiments we used SC2.4.6.2.69232 (not SC2.4.10). The underlying dynamics are sufficiently different that you **cannot** compare runs across the 2 versions!
The `install_sc2.sh` script will install SC2.4.6.2.69232.

## Running experiments
The config file (`src/config/algs/shaq.yaml`) contains default hyper-parameters for SHAQ.
These were changed when running the experiments for the paper (`epsilon_anneal_time = 1000000` for the super-hard maps in SMAC and Predator-Prey).

About the hyperparameter settings of variant experiments, the full details are listed as below.
|  Scenarios | LR for alpha | Epsilon anneal time |
|---|---|---|
| SMAC: 2c_vs_64zg | 0.002 | 50k |
| SMAC: 3s_vs_5z | 0.001 | 50k |
| SMAC: 5m_vs_6m | 0.0005 | 50k |
| SMAC: 6h_vs_8z | 0.0005 | 1mil |
| SMAC: Corridor | 0.0005 | 1mil |
| SMAC: 8m   | 0.0003 | 50k |
| SMAC: 3s5z | 0.0003 | 50k |
| SMAC: 3s5z_vs_3s6z | 0.0003 | 1mil |
| SMAC: 1c3s5z | 0.0002 | 50k |
| SMAC: 10m_vs_11m | 0.0001 | 50k |
| SMAC: MMM2 | 0.0001 | 1mil |
| Predator-Prey | 0.0001 | 1mil |

Please see the Appendix of the paper for the exact hyper-parameters used.

As an example, to run the SHAQ on SMAC: 2c_vs_64zg with epsilon annealed over 50k time steps:
```shell
python3 src/main.py --config=shaq --env-config=sc2 with env_args.map_name=2c_vs_64zg alpha_lr=0.002 epsilon_anneal_time=50000
```

## Citing
If you use part of the work mentioned in this paper, please cite
```
@misc{https://doi.org/10.48550/arxiv.2105.15013,
  doi = {10.48550/ARXIV.2105.15013},
  url = {https://arxiv.org/abs/2105.15013},
  author = {Wang, Jianhong and Zhang, Yuan and Gu, Yunjie and Kim, Tae-Kyun},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Multiagent Systems (cs.MA), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
