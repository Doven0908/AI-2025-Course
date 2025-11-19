# 12组 Agent项目说明文档
## 文件结构及说明
```
group-12/
├── autogen/
├── HumanEval/
|   ├── Scripts/
|   |   ├── custom_tabulate.py
|   |   ├── init_tasks.py
|   |   └── ttsummary.py
|   ├── Tasks
|   ├── Templates
|   └── README.md 
├── Results/
├── dockermod.sh
├── oai.sh
├── OAI_CONFIG_LIST
└── README.md
```
其中autogen文件夹为autogen框架源代码，内含针对autogen的自动化测试框架autogenbench的源代码。HumanEval为代码生成任务，其内部文件夹Scripts为脚本文件夹(custum_tabulate.py用于汇总所有结果生成表格, init_tasks.py用于初始化任务时生成对应的.jsonl文件, ttsummary.py用于计算执行任务总耗时), Tasks文件夹内为所有任务对应的.jsonl文件, 内含所有待测试的样例等信息, Templates文件夹为针对每个任务的模板文件, 代码运行时会通过模板文件和对应样例生成一份真正的可以运行的具体情景文件夹，README.md文件内写了HumanEval任务的使用说明。Results内含了所有任务样例运行结果，以及汇总所有任务运行结果的.csv文件，每个任务的文件夹下也含有汇总所有样例运行结果的.csv文件，主要内容为准确率和运行时间。README.md为大作业说明文档。

## 使用说明
首先在根目录运行下面的命令
```bash
git clone https://github.com/microsoft/autogen.git
pip install -e autogen/samples/tools/autogenbench
```
这样可以根目录内克隆autogen源代码，然后通过本地引入并修改autogenbench测试框架的代码的方式来运行项目。

然后运行下列命令将OPEN_AI的API密钥写入环境变量
```bash
export OAI_CONFIG_LIST=$(cat ./OAI_CONFIG_LIST)
```
或者直接修改任务文件夹下的模板文件夹内的scenario.py文件夹内有关读取密钥的代码，硬编码密钥到该文件中。

由于HumanEval任务文件夹已经克隆在根目录中，可直接运行下列命令来运行任务并导出结果。
```bash
autogenbench run Tasks/human_eval_TwoAgents.jsonl
autogenbench tabulate Results/human_eval_TwoAgents
python HumanEval/Scripts/ttsummary.py
```
其中路径中的任务名可以修改为提供的其他智能体编排对应的任务名。

**注意**:初次运行可能需要根据报错信息安装对应的依赖包，并修改autogen/samples/tools/autogenbench/autogenbench/res/Dockerfile使得Docker初始化时正确引入所需的依赖包。HumanEval/Scripts/ttsummary.py也需要根据运行的具体任务路径进行微小修改（任务结果路径被硬编码到改文件中）。具体来说，修改后的对应位置的Dockerfile内容如下
```Dockerfile
# Host a jsPsych experiment in Azure
FROM python:3.11
MAINTAINER AutoGen

# Upgrade pip
RUN pip install --upgrade pip

# Set the image to the Pacific Timezone
RUN ln -snf /usr/share/zoneinfo/US/Pacific /etc/localtime && echo "US/Pacific" > /etc/timezone

# Pre-load autogen dependencies, but not autogen itself since we'll often want to install the latest from source
RUN pip install autogen-agentchat[teachable,lmm,graphs,websurfer]~=0.2
RUN pip uninstall --yes autogen-agentchat~=0.2

# Pre-load popular packages as per https://learnpython.com/blog/most-popular-python-packages/
RUN pip install numpy pandas matplotlib seaborn scikit-learn requests urllib3 nltk pillow pytest

RUN pip install autogen~=0.2
RUN pip install ag2[openai]
```

## 运行结果
详见Results文件夹下csv格式文件以及其中的每个任务对应文件夹下csv格式文件。
