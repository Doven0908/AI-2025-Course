# 足式机器人行走
# 任务概述
在仿真环境训练四足或人形步态，训练四足机器人平地行走，对比了Stable-Baselines3（ PPO/SAC 等）与 RSL-RL 高性能 PPO）两种工作流；获得了稳定行走与抗扰动能力，并可用tensorboard查看训练结果，查看收敛速度与步态稳定性。
AMP-RSL-RL 做运动先验模仿，并进行重定向后做了仿真。
在复杂地面进行了四足机器人与人形机器人的训练，并在仿真中进行了验证。
# 使用说明
## 安装Isaac lab
-[Isaaclab安装](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).此处建议使用conda安装，因为它简化了从终端调用Python脚本的操作。
## 1.使用stable-baselines3进行强化学习训练A1机器人
- 安装python模块 (for stable-baselines3)
 ```bash
./isaaclab.sh -i sb3
```
- 训练
 ```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0 --headless
```
- 运行测试
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0  --num_envs 32 --checkpoint /logs/sb3/Isaac-Velocity-Flat-Unitree-A1-v0/XXX/model_xxx.zip
```
- Demo


https://github.com/user-attachments/assets/def9eb68-2b9f-4140-b821-069861d79e7a


## 2.使用RSL-RL进行强化学习训练A1机器人
- 安装python模块 (for rsl-rl)
```bash
./isaaclab.sh -i rsl_rl
```
- 训练
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-A1-v0  --headless
``` 
- 运行测试
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Unitree-A1-v0  --num_envs 32 --load_run run_folder_name --checkpoint /logs/rsl_rl/unitree_a1_flat/XXX/model_xxx.pt
```
- Demo


https://github.com/user-attachments/assets/5505405c-07f5-4fe9-a590-8f98964b0ca3


## 3.复杂地形
### 四足
- 训练
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go2-v0  --headless
```
- 运行测试
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-Unitree-Go2-Play-v0  --num_envs 32 --load_run run_folder_name --checkpoint /rsl_rl/unitree_go2_rough/XXX/model_xxx.pt
```
- Demo


https://github.com/user-attachments/assets/c0e7d9ab-8340-4dd0-a038-942c04129812


### 人形
- 训练
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-G1-v0 --headless
```
- 运行测试
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-G1-Play-v0  --num_envs 32 --load_run run_folder_name --checkpoint /logs/rsl_rl/g1_rough/XXX/model_xxx.pt
```
- Demo


https://github.com/user-attachments/assets/00a78609-d79d-4026-ad49-95e3b79c4403


## AMP
### 安装
- 安装IsaacLab 按照文中开头所述操作即可
- 克隆这个仓库要和Isaac Lab的安装分开进行，即要在IsaacLab目录之外进行
 ```bash
# Option 1: HTTPS
git clone https://github.com/zitongbai/legged_lab

# Option 2: SSH
git clone git@github.com:zitongbai/legged_lab.git

cd legged_lab
git checkout v2.2.1
```
- 使用已经安装了Isaac Lab的Python解释器来安装这个库
```bash
python -m pip install -e source/legged_lab
```
- 克隆RSL-RL库，使其与Isaac Lab安装和legged_lab仓库分开即在IsaacLab和legged_lab之外
```bash
# Option 1: HTTPS
git clone https://github.com/zitongbai/rsl_rl.git
# Option 2: SSH
git clone git@github.com:zitongbai/rsl_rl.git

cd rsl_rl
git checkout feature/amp_3
```
- 安装RSL_RL库
```bash
python -m pip install -e .
```
- 在legged lab中下载Unitree G1的USD模型
```bash
gdown "https://drive.google.com/drive/folders/1rlhZcurMenq4RGojZVUDl3a6Ja2hm-di?usp=drive_link" --folder -O ./source/legged_lab/legged_lab/data/Robots/
```
- 在legged lab中下载Unitree G1的动捕数据
```bash
gdown "https://drive.google.com/drive/folders/1tXtyjgM_wwqWNwnpn8ny5b4q1c-GxZkm?usp=sharing" --folder -O ./source/legged_lab/legged_lab/data/
```
- 通过运行以下命令来验证该扩展是否正确安装
```bash
python scripts/rsl_rl/train.py --task=LeggedLab-Isaac-Velocity-Rough-G1-v0 --headless
```
### 训练
- 训练
```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 --headless --max_iterations 10000
```
- 在非默认的GPU上进行训练
```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-Flat-G1-v0 --headless --max_iterations 50000 --device cuda:x agent.device=cuda:x
```
### 运行测试
-运行
```bash
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-Flat-G1-Play-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```
- Demo


https://github.com/user-attachments/assets/c5f40eb8-bb46-4947-8b58-da07e7b857e2

