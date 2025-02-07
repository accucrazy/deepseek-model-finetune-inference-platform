import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomStoppingCriteria(LogitsProcessor):
    """自定義邏輯來控制token生成"""
    def __init__(self, tokenizer, stop_token_ids):
        self.tokenizer = tokenizer
        self.stop_token_ids = stop_token_ids
        self.generated_tokens = []

    def __call__(self, input_ids, scores):
        # 大幅降低特定token的生成機率
        for token_id in self.stop_token_ids:
            scores[:, token_id] = -float('inf')
        return scores

def load_model(model_path):
    """載入微調後的模型和tokenizer"""
    logger.info(f"Loading model from {model_path}")
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # 載入原始模型的tokenizer
        base_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        logger.info(f"Loaded tokenizer from {base_model_name}")
        
        # 載入微調後的模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            # 不修改任何配置，使用模型原有的設置
        )
        logger.info("Model loaded successfully")
        
        # 設置為評估模式
        model.eval()
        
        # 確保有正確的特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def generate_response(model, tokenizer, question, temperature=0.9, top_p=0.9, max_new_tokens=100, repetition_penalty=1.8):
    """生成回應"""
    try:
        # 使用與訓練數據相同的格式
        prompt = f"問題：{question}\n答案："
        logger.info(f"Input prompt: {prompt}")
        
        # Tokenize輸入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_length = inputs["input_ids"].shape[1]
        
        # 移動到GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        # 設置為評估模式
        model.eval()
        
        # 準備停止詞token
        stop_words = ["問題：", "\n", "第"]
        stop_token_ids = []
        for word in stop_words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            stop_token_ids.extend(ids)
        
        # 設置logits processor
        logits_processor = CustomStoppingCriteria(tokenizer, stop_token_ids)
        
        # 生成回應
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # 使用參數控制
                min_new_tokens=5,   # 確保至少生成一定長度
                num_return_sequences=1,
                do_sample=True,    # 使用採樣
                temperature=temperature,    # 使用參數控制
                top_p=top_p,         # 使用參數控制
                num_beams=3,       # 使用束搜索
                repetition_penalty=repetition_penalty,  # 使用參數控制
                logits_processor=[logits_processor],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解碼並處理回應
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # 清理回應
        response = response.strip()
        
        # 在任何停止詞處截斷
        for stop_word in stop_words:
            if stop_word in response:
                response = response.split(stop_word)[0]
        
        # 驗證答案
        if not response or len(response) < 2:
            return "無法生成有效回答，請重試"
            
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"生成過程中發生錯誤：{str(e)}"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with fine-tuned model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the fine-tuned model')
    parser.add_argument('--question', type=str, required=True,
                      help='Question to ask the model')
    
    args = parser.parse_args()
    
    # 載入模型
    model, tokenizer = load_model(args.model_path)
    
    # 生成回應
    answer = generate_response(model, tokenizer, args.question)
    print(f"\n問題：{args.question}")
    print(f"答案：{answer}") 