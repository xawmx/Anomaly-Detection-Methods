# 🚀 Blockchain Transaction Behavior Regulation Research  
**区块链交易行为监管技术研究**

<div align="right">
    <a href="#readme-cn">🇨🇳 中文</a> | <a href="#readme-en">🇺🇸 English</a>
</div>

---

## 📖 项目概述 | Project Overview

### 中文
本项目旨在探索和实现区块链交易行为的有效监管方法。通过对区块链交易数据的分析与建模，识别异常交易行为，保障网络的安全和稳定。研究成果可以为区块链交易行为的监管提供技术支持。

### English
This project aims to explore and implement effective methods for regulating blockchain transaction behaviors. By analyzing and modeling blockchain transaction data, it identifies abnormal behaviors to ensure the security and stability of the network. The results of this research provide technical support for blockchain transaction regulation.

---

## 📂 项目结构 | Project Structure

```text
code_tracking_eth/  
├── graph.py           # 图操作相关功能，如边映射和节点采样
├── model.py           # 模型定义，包括TGNN类和Transformer模型
├── data_load.py       # 数据加载与处理
├── utils.py           # 工具函数，如sigmoid
code_ad_eth/  
├── preprocessing.py   # 数据预处理操作
├── weight_choice.py   # 权重选择逻辑