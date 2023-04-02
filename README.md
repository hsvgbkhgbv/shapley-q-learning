# SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning

This is the implementation of the paper SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning (https://arxiv.org/abs/2105.15013), published on NeurIPS 2022.

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

## Visualizing the learned values
We also provide the method to visualize the learned values during test. The details are as follows:
1. Set the param `evaluate` as `True` and set an address for saving the testing result in string to `save_batch_path` in the `default.yaml`. 

2. Set the checkpoint address to `checkpoint_path` that saves the model you would test in `default.yaml`.

3. Set the trajectory during test saving address to `save_batch_path` in `default.yaml`.

4. If you would like to test with the greedy policies, you need to set `epsilon_test` as `False`. Otherwise, you need to set `epsilon_test` as a float from `0` to `1.` to indicate the probability of performing random actions.

5. Run the following example command:
```shell
python3 src/main.py --config=shaq --env-config=pred_prey_punish with env_args.miscapture_punishment=-1 checkpoint_path=results/models/shaq__2022-07-31_11-31-44 evaluate=True epsilon_test=False save_batch_path=[...]
```

6. Remove the `save_batch_path` and set the trajectory saving path to `load_batch_path`.

7. Set the paths for saving values, actions and state to `save_values_path`, `save_actions_path` and `save_state_path` respectively. These are saved in pickle files.

8. Run the following example command:
```shell
python3 src/main.py --config=shaq --env-config=pred_prey_punish with env_args.miscapture_punishment=-1 checkpoint_path=results/models/shaq__2022-07-31_11-31-44 evaluate=True epsilon_test=False load_batch_path=[...] save_values_path=[...] save_actions_path=[...] save_state_path=[...]
```

9. You can visualize and analyze the learned values, actions and states through the storage in the saved pickle files.

## Citing
If you use part of the work mentioned in this paper, please cite
```
@article{wang2022shaq,
  title={SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning},
  author={Wang, Jianhong and Zhang, Yuan and Gu, Yunjie and Kim, Tae-Kyun},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={5941--5954},
  year={2022}
}
```
