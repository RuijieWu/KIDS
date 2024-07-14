<!--
 * @Date: 2024-07-14 13:11:11
 * @LastEditTime: 2024-07-14 14:49:57
 * @Description: 
-->
# KIDS Engine

## Layout

```plaintext
KIDS Engine
├─artifact
│  ├─graph_list
│  ├─graph_visual
│  └─models
├─ Dataset
├─ KAIROS
├─ Log
├─ Preprocess
├─ analyser.py
├─ cadets_e3_process.py
├─ config.py
├─ embedder.py
├─ engine.py
├─ investigator.py
├─ model.py
├─ show_output.py
├─ utils.py
└─ README.md
```

- artifact 文件夹包含了 KIDS Engine 的输出，其中
  - graph_list 文件夹用于存放 KIDS Engine 分析计算得到的时间窗口文件，每一个时间窗口文件内记录了这些时间窗口内的事件及其异常值信息
  - graph_visual 文件夹用于存放 KIDS Engine 针对针对异常行为与攻击行为生成的溯源图
  - model 文件夹用于存放 KIDS Engine 需要使用的 .pt 模型文件
- Dataset 文件夹用于存放数据集的原始数据
- KAIROS 文件夹内包含了基于 Kairos 模型 Demo 代码编写的从数据处理到模型训练再到异常调查的流水线
- Log 文件夹用于存放 KIDS Engine 运行过程中生成的日志文件
- Preprocess 文件夹包含了对 DARPA 组织提供的各类数据集进行数据处理与模型训练所需的代码
- analyser.py 文件是 KIDS Engine 的分析模块，负责对数据存储模块中存储的系统日志进行分析
- cadets_e3_process.py 文件是对 cadets e3 数据集进行数据处理的脚本，KIDS 系统在演示时采用该数据集
- config.py 文件存储了 KIDS Engine 运行时的相关参数
- embedder.py 文件会将数据存储模块中的系统日志记录嵌入为图神经网络中的结点以便 KIDS Engine 进行分析与预测
- engine.py 文件是 KIDS Engine 的主程序，它会自行加载模型数据并控制其它模块进行入侵检测工作
- investigator.py 文件是 KIDS Engine 的调查模块，在图神经网络计算出异常值后超出异常值阈值的时间窗口进行攻击调查
- model.py 文件为 KIDS Engine 的模型定义文件
- show_output.py 文件是查看 KIDS Engine 输出的脚本程序，它也可以把数据库内的输出导出为 csv 文件方便数据迁移与调试
- utils.py 文件内存放了 KIDS Engine 各模块通用的功能组件

## Run cadets e3 Demo

在 KIDS DataStorage 设置完毕后，提取 cadets e3 数据集中的数据并插入到数据库中

```bash
python cadets_e3_process.py
```

运行 KIDS Engine

```bash
python engine.py run -begin 2018-04-06 00:00:00 -end 2018-04-07 00:00:00
```

查看数据库输出(并导出为 CSV)

```bash
python show_output.py
ls ./Log
```
