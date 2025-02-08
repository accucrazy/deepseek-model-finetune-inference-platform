# The Pocket Company - Deepseek 模型調教與推論平台

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

一個用於 fine-tune 和推理 Deepseek 模型的簡單圖形化UI。


## 系統需求

- Python 3.10 或以上
- CUDA 12.1 或以上
- GPU 顯卡 VRAM >= 16GB (使用 1.3B 模型)
- GPU 顯卡 VRAM >= 32GB (使用 6.7B 模型)

## 支援模型版本

- deepseek-ai/deepseek-coder-1.3b-base
- deepseek-ai/deepseek-coder-6.7b-base
- deepseek-ai/deepseek-coder-33b-base (需要更大 VRAM)

## 依賴套件版本

- PyTorch >= 2.0.0 (CUDA 12.1 版本)
- Transformers >= 4.36.0
- Streamlit >= 1.24.0
- Accelerate >= 0.25.0
- Bitsandbytes >= 0.41.1
- Safetensors >= 0.4.0

## 功能特點

- 🚀 簡潔的網頁界面
- 💾 支持 JSON/JSONL 格式的訓練數據
- ⚡ 實時顯示 GPU 狀態
- 📊 可視化訓練過程
- 🔮 模型推理與對話

## 安裝

1. 創建 conda 環境：
```bash
conda create -n deepseek python=3.10
conda activate deepseek
```

2. 安裝 PyTorch (CUDA 12.1)：
```bash
pip install -r requirements-torch.txt
```

3. 安裝其他依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

### 訓練模型

1. 準備數據集（JSONL格式）：
```json
{"text": "問題：你的問題\n答案：對應的答案"}
```

2. 運行訓練：
```bash
python train.py --model_name "deepseek-ai/deepseek-coder-1.3b-base" --dataset_path "your_dataset.jsonl" --output_dir "fine_tuned_model"
```

### 啟動界面

```bash
streamlit run app.py
```

## 數據集格式

支持兩種格式：
 JSONL格式（推薦）：
```json
{"text": "問題：問題1\n答案：答案1"}
{"text": "問題：問題2\n答案：答案2"}
```



## 目錄結構

```
fine_tune_deepseek/
├── app.py              # 主界面
├── train.py            # 訓練腳本
├── inference.py        # 推理腳本
├── convert_data.py     # 數據轉換工具
├── requirements.txt    # 依賴包
├── requirements-torch.txt  # PyTorch 依賴
├── qa_dataset.jsonl    # 示例數據集
├── LICENSE            # MIT 授權條款
└── README.md          # 說明文檔
```

## 注意事項

- 確保有足夠的 GPU VRAM (1.3B 模型建議 16GB 以上)
- 建議使用 NVIDIA 顯卡驅動 >= 525.60.11
- 建議使用 CUDA 12.1 或以上版本
- 首次加載模型可能需要一些時間
- 訓練時的 batch_size 需根據 VRAM 大小調整：
  - 16GB VRAM: batch_size = 4
  - 24GB VRAM: batch_size = 6
  - 32GB VRAM: batch_size = 8
- 推理參數預設值：
  - Temperature: 0.6
  - Top P: 0.9
  - Maximum New Tokens: 100
  - Repetition Penalty: 1.9

## 作者

🏢 The Pocket Company by Accucrazy 肖準 
CEO Ian Wu - 商學院社會組用Cursor搞出來的
