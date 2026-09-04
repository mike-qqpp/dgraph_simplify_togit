ta_split/` - 数据集划分脚本，将数据集划分为 train 和 test，test 用于推理

### Ours 代码（先制作特征）

#### code_cross
原始特征制作和模型训练：
- `run_all.sh` - 制作原始特征
- `run_trainmodel.sh` - 训练模型

#### code_cross_tfinance
TFinance 数据集专用（独立的 prompt 设计代码）：
- `run_fe_train.sh` - 制作训练集原始特征
- `run_fe_test.sh` - 制作测试集原始特征
- `run_train.sh` - 训练模型

> 注：TFinance 采用独立的代码，因与其他三个数据集使用同一代码生成的特征效果不佳。

#### code_cross_all
交叉组合特征实验代码（各 `.py` 文件）

### 消融实验代码（CPU 版本）
各 `*_split` 目录包含将训练和推理分开的各个算法的代码：

| 目录 | 推理脚本 |
|------|----------|
| `AMNet_split/` | `run_infer_cpu.sh` |
| `DGA-GNN_split/` | `./code/run_infer_cpu.sh` |
| `DSGAD-mul_split/` | `run_infer_cpu.sh` |
| `GHRN_split/` | `run_infer_cpu.sh` |
| `antifraud_split/` | `run_infer_cpu.sh` |
