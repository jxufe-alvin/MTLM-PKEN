# MTLM-PKEN(Multi-Task Learning Model Integrating Prior Knowledge and Expert Networks)模型源码
论文"融合先验知识和专家网络的多任务心理疾病识别"的源代码

## 数据集说明(dataset)
论文数据集来源请参考论文"融合先验知识和专家网络的多任务心理疾病识别"。
受限于伦理要求，请向数据集来源论文作者(组织)申请获取原始数据。

## 运行(RUN)
```
sh run_exp.sh
```
## 环境(Environment)
```
pip install -r requirements.txt
```

## 说明与致谢(Instructions and Acknowledgments)
- 本项目部分代码基于[LibMTL](https://gitcode.com/gh_mirrors/li/LibMTL/?utm_source=artical_gitcode&index=top&type=card&webUrl)进行修改， LibMTL 是一个基于 PyTorch 构建的多任务学习（Multi-Task Learning，MTL）开源库，特此感谢！
- MTLM-PKEN/LibMTL目录中含有CGC、MMOE等对比模型代码。
- LLMsEvaluation目录: LLMs_api_pred_smdd.py代码支持GPT系列、GLM-4、DeepSeek、QWEN系列等大模型在SMDD上的评估。

### 引用(Cite)
如果您使用了该源码，请在CNKI上检索并引用以下论文:
```
融合先验知识和专家网络的多任务心理疾病识别(刘德喜等)
```