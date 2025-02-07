import os
import sys
import logging
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# First check PyTorch installation
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

try:
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import load_dataset
except ImportError as e:
    logger.error(f"Error importing required packages: {str(e)}")
    raise

from packaging import version

# Check PyTorch version
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
else:
    raise ImportError("PyTorch version >= 2.0.0 is required")

def diagnose_cuda():
    """診斷 CUDA 環境"""
    logger.info("=== CUDA 環境診斷 ===")
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 是否可用 (torch.cuda.is_available): {torch.cuda.is_available()}")
    logger.info(f"CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    if torch.cuda.is_available():
        logger.info(f"CUDA 設備數量: {torch.cuda.device_count()}")
        logger.info(f"當前 CUDA 設備: {torch.cuda.current_device()}")
        logger.info(f"CUDA 設備名稱: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA 診斷信息:")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        logger.info(f"PATH: {os.environ.get('PATH', 'Not set')}")
    logger.info("==================")

def setup_cuda():
    """設置 CUDA 環境"""
    if torch.cuda.is_available():
        # 設置默認設備為 CUDA
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
        logger.info(f"成功設置 CUDA 設備: {device}")
        return device
    else:
        logger.warning("CUDA 不可用，將使用 CPU")
        return torch.device("cpu")

def check_gpu_status():
    """檢查 GPU 狀態"""
    logger.info("=== GPU 狀態檢查 ===")
    logger.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU 數量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"當前 GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU 記憶體使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.info(f"GPU 記憶體快取: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    logger.info("==================")

def load_model_and_tokenizer(model_name):
    """載入預訓練模型和分詞器"""
    logger.info(f"Loading model and tokenizer: {model_name}")
    
    # 診斷 CUDA 環境
    diagnose_cuda()
    
    # 設置 CUDA 環境
    device = setup_cuda()
    
    # 檢查 GPU 狀態
    check_gpu_status()
    
    # 添加詳細的載入信息
    logger.info("開始載入模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float32
        )
        logger.info(f"模型載入完成，模型設備: {next(model.parameters()).device}")
    except Exception as e:
        logger.error(f"模型載入失敗: {str(e)}")
        raise
    
    logger.info("開始載入 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info("Tokenizer 載入完成")
    
    # 確保 tokenizer 有 pad token
    if tokenizer.pad_token is None:
        logger.info("Pad token 不存在，嘗試設定 pad token...")
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            logger.info("使用 eos_token 作為 pad token")
        else:
            logger.info("eos_token 也不存在，新增自定義 pad token")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
            logger.info(f"設定 pad token 為 [PAD]，ID: {model.config.pad_token_id}")
    
    return model, tokenizer

def prepare_dataset(dataset_path, tokenizer, max_length=512):
    """準備和預處理數據集"""
    logger.info(f"=== Starting Dataset Preparation ===")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Max length: {max_length}")
    
    try:
        # 檢查文件是否存在和內容
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        # 讀取文件內容進行驗證
        logger.info("=== Reading Dataset File ===")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Raw content length: {len(content)}")
            logger.info(f"Content preview: {content[:200]}")
        
        # 直接使用 datasets 庫載入 JSONL 文件
        logger.info("=== Loading Dataset with Huggingface ===")
        dataset = load_dataset('json', data_files={'train': dataset_path})
        logger.info(f"Dataset loaded successfully with {len(dataset['train'])} examples")
        
        # 檢查第一個樣本
        logger.info("=== Checking First Example ===")
        first_example = dataset['train'][0]
        logger.info(f"First example type: {type(first_example)}")
        logger.info(f"First example content: {first_example}")
        
        def tokenize_function(examples):
            """將文本轉換為 token"""
            try:
                logger.info("=== Starting Tokenization Function ===")
                
                # 確保我們有文本數據
                if 'text' not in examples:
                    available_keys = list(examples.keys())
                    logger.error(f"Missing 'text' field. Available keys: {available_keys}")
                    raise ValueError(f"Missing 'text' field. Available keys: {available_keys}")
                
                texts = examples['text']
                logger.info(f"Raw texts type: {type(texts)}")
                logger.info(f"Raw texts content: {texts}")
                
                # 確保文本是列表格式
                if isinstance(texts, str):
                    texts = [texts]
                elif not isinstance(texts, list):
                    texts = [str(texts)]
                
                # 過濾並清理文本
                valid_texts = []
                for i, text in enumerate(texts):
                    if text is None:
                        logger.warning(f"Text {i} is None, skipping")
                        continue
                        
                    if not isinstance(text, str):
                        try:
                            text = str(text)
                            logger.info(f"Converted text {i} to string")
                        except:
                            logger.warning(f"Could not convert text {i} to string")
                            continue
                    
                    text = text.strip()
                    if text.lower() == "none":
                        logger.warning(f"Text {i} is the string 'None' after stripping, skipping")
                        continue
                    
                    if text:
                        valid_texts.append(text)
                        logger.info(f"Added valid text {i}: {text[:100]}...")
                
                if not valid_texts:
                    logger.error("No valid texts found after filtering")
                    raise ValueError("No valid texts found in batch")
                
                # 執行分詞
                logger.info(f"Tokenizing {len(valid_texts)} texts")
                tokenized = tokenizer(
                    valid_texts,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors=None  # 確保返回的是Python列表而不是張量
                )
                
                # 驗證 attention_mask 是否存在
                if 'attention_mask' not in tokenized or tokenized['attention_mask'] is None:
                    logger.error("Tokenization did not produce attention_mask")
                    raise ValueError("Tokenization error: no attention_mask")
                
                # 驗證分詞結果
                if not tokenized or 'input_ids' not in tokenized:
                    logger.error("Tokenization failed to produce input_ids")
                    logger.error(f"Tokenized output: {tokenized}")
                    raise ValueError("Tokenization failed")
                
                logger.info(f"Tokenization successful. Keys: {tokenized.keys()}")
                logger.info(f"Input IDs shape: {len(tokenized['input_ids'])}x{len(tokenized['input_ids'][0])}")
                
                # 驗證每個 tokenized sample 是否為 None
                for idx, ids in enumerate(tokenized['input_ids']):
                    if ids is None:
                        logger.error(f"Tokenization produced None for input_ids at index {idx}")
                        raise ValueError("Tokenization error: None in input_ids")
                
                return tokenized
                
            except Exception as e:
                logger.error(f"Error in tokenize_function: {str(e)}")
                logger.error("Full error details:", exc_info=True)
                raise
        
        # 對數據集進行分詞處理
        logger.info("=== Starting Dataset Mapping ===")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
            batch_size=1,
            desc="Tokenizing dataset"
        )
        
        # 驗證處理後的數據集
        logger.info("=== Validating Tokenized Dataset ===")
        if not tokenized_dataset or 'train' not in tokenized_dataset:
            logger.error("Failed to create tokenized dataset")
            raise ValueError("Dataset tokenization failed")
            
        logger.info(f"Tokenized dataset size: {len(tokenized_dataset['train'])}")
        logger.info(f"Tokenized features: {tokenized_dataset['train'].features}")
        
        # 檢查第一個處理後的樣本
        first_tokenized = tokenized_dataset['train'][0]
        logger.info(f"First tokenized example: {first_tokenized}")
        
        return tokenized_dataset
        
    except Exception as e:
        logger.error("=== Dataset Preparation Error ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        raise

class SafeDataCollator:
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.original_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)
        
    def __call__(self, examples):
        """安全的數據整理函數，處理可能的None值"""
        try:
            logger.info("=== Data Collator Processing ===")
            logger.info(f"Received {len(examples)} examples")
            
            # 驗證輸入
            if not examples:
                raise ValueError("Empty examples list")
                
            # 檢查每個樣本
            valid_examples = []
            for i, example in enumerate(examples):
                logger.info(f"Processing example {i}:")
                logger.info(f"  Type: {type(example)}")
                logger.info(f"  Content: {example}")
                
                if example is None:
                    logger.warning(f"  Example {i} is None, skipping")
                    continue
                    
                if not isinstance(example, dict):
                    logger.warning(f"  Example {i} is not a dictionary, skipping")
                    continue
                    
                if 'input_ids' not in example:
                    logger.warning(f"  Example {i} missing input_ids, skipping")
                    continue
                    
                valid_examples.append(example)
            
            if not valid_examples:
                raise ValueError("No valid examples found after filtering")
                
            logger.info(f"Processing {len(valid_examples)} valid examples")
            
            # 在使用原始 collator 前，新增每個有效例子的 input_ids 詳細日誌
            for i, example in enumerate(valid_examples):
                input_ids = example.get('input_ids', [])
                logger.info(f"Example {i} input_ids: {input_ids}, token types: {[type(token) for token in input_ids]}")

            # 使用原始的collator處理有效樣本
            result = self.original_collator(valid_examples)
            logger.info(f"Collation successful. Output keys: {result.keys()}")
            return result
            
        except Exception as e:
            logger.error("=== Data Collator Error ===")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Full error details:", exc_info=True)
            raise

def collate_fn(examples, tokenizer):
    """數據整理函數"""
    batch = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )(examples)
    return batch

def train(
    model_name,
    dataset_path,
    output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_length=256,
    warmup_steps=100
):
    """訓練模型的主要函數"""
    
    logger.info("=== Starting Training Process ===")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Set device first
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Set all random seeds for reproducibility
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # 載入模型和分詞器
        logger.info("=== Loading Model and Tokenizer ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model is None or tokenizer is None:
            raise ValueError("Failed to load model or tokenizer")
            
        # 驗證tokenizer設置
        logger.info("=== Validating Tokenizer Settings ===")
        if tokenizer.pad_token is None:
            logger.info("Setting pad_token to eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            
        # 準備數據集
        logger.info("=== Preparing Dataset ===")
        dataset = prepare_dataset(dataset_path, tokenizer, max_length)
        
        if dataset is None:
            raise ValueError("Dataset preparation failed")
            
        # 驗證數據集
        logger.info("=== Validating Dataset ===")
        if 'train' not in dataset:
            raise ValueError("Dataset does not contain 'train' split")
            
        train_dataset = dataset['train']
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # Convert dataset to torch format and add labels
        def add_labels(example):
            example["labels"] = example["input_ids"].copy()
            return example
            
        logger.info("=== Converting Dataset Format ===")
        train_dataset = train_dataset.map(add_labels)
        
        # 設定訓練參數
        logger.info("=== Setting up Training Arguments ===")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=1,
            save_strategy="epoch",
            fp16=False,
            bf16=False,
            optim="adamw_torch",
            gradient_checkpointing=True,
            no_cuda=not torch.cuda.is_available(),
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            seed=seed,
            data_seed=seed,
            dataloader_num_workers=0,
            remove_unused_columns=True,
            group_by_length=True,
            auto_find_batch_size=True,
            full_determinism=True,
            torch_compile=False,
            include_inputs_for_metrics=True,
            max_grad_norm=1.0,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        logger.info(f"Training arguments:\n{training_args}")
        
        # 初始化訓練器
        logger.info("=== Initializing Trainer ===")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=SafeDataCollator(tokenizer, mlm=False),
        )
        
        if trainer is None:
            raise ValueError("Failed to create trainer")
        
        # 開始訓練
        logger.info("=== Starting Training ===")
        trainer.train()
        
        # 儲存模型
        logger.info("=== Saving Model ===")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("=== Training Completed Successfully ===")
        
        return trainer
        
    except Exception as e:
        logger.error("=== Training Error ===")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full error details:", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune DeepSeek model')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name or path of the pre-trained model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to the dataset file (JSONL format)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=3,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Training batch size per device')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                      help='Maximum sequence length')
    
    args = parser.parse_args()
    
    train(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    ) 