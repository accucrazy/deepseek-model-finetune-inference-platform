import json
import os

def convert_to_jsonl(input_file, output_file):
    """將訓練數據轉換為JSONL格式"""
    # 讀取輸入的 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 將數據轉換為 JSONL 格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data['train']:
            if 'messages' in item:
                # 找到助手的回應
                assistant_messages = [msg['content'] for msg in item['messages'] 
                                   if msg['role'] == 'assistant']
                if assistant_messages:
                    # 使用最後一個助手回應作為訓練文本
                    text = assistant_messages[-1]
                    json_line = {'text': text}
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # 使用示例數據集
    input_file = 'example_dataset.json'
    output_file = 'temp_datasets/converted_dataset.jsonl'
    
    # 確保輸出目錄存在
    os.makedirs('temp_datasets', exist_ok=True)
    
    convert_to_jsonl(input_file, output_file)
    print(f"數據已轉換並保存到 {output_file}") 