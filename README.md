# 中文场景文字识别
## 功能
- 文字方向检测（VGG16）
- 文本区域检测（CTPN）
- 文本内容识别（DenseNet + CTC）
## 思路（no augmentation）
- 思路一
  - Densenet + CTC（CTC Loss的提出为OCR的端到端模型提供巨大的可能性）
- 思路二
  - DenseNet + BLSTM + CTC（双向LSTM作为语言模型，进一步强化自然语言处理的捕捉*效果不明显*）