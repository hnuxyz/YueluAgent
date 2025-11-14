# 麓言湘语：以InternLM之智，传芷兰之道

<p align="center">
<img src="https://img.shields.io/badge/python-3.10-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.19-5D91D4.svg"></a>
<a href="https://github.com/modelscope/ms-swift"><img src="https://img.shields.io/badge/ms%20swift-3.6-green"></a>
</p>

![](src\Fig\logo.png)

##  📖 目录

- [简介](#-简介)
- [环境配置](#-环境配置)
- [微调与部署](#-微调与部署)
- [License](#-license)

## 📝 简介
麓言湘语是一个基于岳麓书院传统文化知识库构建的AI智能体，致力于岳麓书院传统文化的数字化传承与智能化解读。本项目依托于湖南大学[“书院文化传承微型智能体设计大赛”](https://mp.weixin.qq.com/s/A2yFtT2eLHnKpAJ0nrvVdg?scene=1&click_id=7)与上海AI Lab[书生·浦语第六期大模型实战营](https://aicarrier.feishu.cn/wiki/VC12w2b7IiyZXlkfuxzch8K6n8f)，基于InternLM3-8B模型进行LoRA plus微调，结合[ModelScope Swift](https://github.com/modelscope/ms-swift)全链框架优化，微调数据集通过ChatGPT-4o与DeepSeek生成并经人工精校。采用RAG增强检索技术，借助[阿里百炼大模型平台](https://bailian.console.aliyun.com/)Text-Embedding v4 API构建语义向量，精准链接湖南大学岳麓书院讲习团讲解词，实现知识的深度融合与动态响应。前端通过LangChain与Gradio搭建，实现用户友好交互体验，为传统文化注入现代智能解读力量。

![](src\Fig\pipeline.png)

## 🛠️ 环境配置
### PEFT环境(ms-swift环境配置)

```bash
conda create -n ms-swift python=3.10
conda activate ms-swift
pip install ms-swift==3.6
```

### 部署环境

```
conda create -n Agent python=3.10
conda activate Agent
pip install -r requirements.txt
```


## 🚀 微调与部署

- 下载PEFT数据集和RAG文本数据库，链接；
- 下载InternLM3-8B模型权重；
- 进入阿里百炼大模型平台，申请API，将API-Key等相关信息填入YueLu_Agent.py中；
- 若有开源之心，还需获取modelscope的访问令牌，并填入upload.py中。

### PEFT微调

- 将相关数据及基座模型配置好后，命令行进入src文件夹路径，即可在ms-swift中一键启动：

```bash
conda activate ms-swift
bash sft.sh
```

- 微调完成后，选择想要合并权重的checkpoint，完成合并：

```bash
bash merge.sh
```

- 开源权重只需将modelscope访问令牌和合并后的权重路径写入upload.py文件并填写相关信息即可：

```
python upload.py
```

### 启动Web UI界面

- 本项目部署在上海AI Lab提供的A100工作站上，因此使用VS code或其他工具进行端口转发Gradio界面，实现快速前端交互：

```bash
conda activate Agent
VS code运行YueLu_Agent.py
```

- 使用VS code端口转发功能，默认是127.0.0.1:7860，端口号可自行在YueLu_Agent.py中修改合法值；
- 在本地浏览器中输入127.0.0.1:7860（根据自己部署的情况修改）
- 交互：

![](src\Fig\UI.png)


## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License！
