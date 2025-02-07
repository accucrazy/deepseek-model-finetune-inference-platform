import streamlit as st
import json
import os
import torch
import plotly.graph_objects as go
from datetime import datetime
import glob
from inference import generate_response, load_model

# 设置页面配置
st.set_page_config(
    page_title="The Pocket Company",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #FF1493;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #FF1493;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .platform-name {
        font-size: 1.2rem;
        color: #FF1493;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .company-signature {
        font-size: 0.9rem;
        color: #FF1493;
        text-align: right;
        margin-top: 1rem;
        padding-right: 1rem;
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 5px 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# 主标题
st.markdown('<h1 class="main-header">🏢 The Pocket Company by Accucrazy 肖準</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">🤖 Deepseek模型調教與推論平台</h2>', unsafe_allow_html=True)
st.markdown('<h3 class="platform-name">🚀 Deepseek Model Fine-tuning and Inference Platform</h3>', unsafe_allow_html=True)

# 添加固定的品牌标识
st.markdown('<p class="company-signature">🏢 The Pocket Company by Accucrazy 肖準</p>', unsafe_allow_html=True)

# 顯示 CUDA 狀態
cuda_status = st.empty()
with cuda_status.container():
    st.subheader("💻 系統狀態")
    col1, col2 = st.columns(2)
    
    with col1:
        if torch.cuda.is_available():
            st.success("✨ CUDA 可用")
            st.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.error("❌ CUDA 不可用")
            st.warning("⚠️ 請確保：\n1. 已安裝 NVIDIA 驅動\n2. CUDA 工具包已正確安裝\n3. PyTorch 已安裝 CUDA 版本")
    
    with col2:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            st.info(f"📊 GPU 記憶體使用: {memory_allocated:.2f} GB")
            st.info(f"💾 GPU 記憶體保留: {memory_reserved:.2f} GB")

# 创建两个标签页
tab1, tab2 = st.tabs(["⚡ 模型訓練", "🔮 模型推理"])

with tab1:
    st.header("⚡ 模型訓練")
    
    # 側邊欄配置
    with st.sidebar:
        st.header("⚙️ 訓練配置")
        
        # 模型选择
        model_name = st.selectbox(
            "🤖 選擇基礎模型",
            ["deepseek-ai/deepseek-coder-1.3b-base",
             "deepseek-ai/deepseek-coder-6.7b-base",
             "deepseek-ai/deepseek-coder-33b-base"],
            help="選擇要 fine-tune 的基礎模型"
        )
        
        # 训练参数设置
        st.subheader("📊 訓練參數")
        col1, col2 = st.columns(2)
        with col1:
            num_epochs = st.number_input("🔄 訓練輪數", min_value=1, value=3)
            batch_size = st.number_input("📦 Batch Size", min_value=1, value=4)
        with col2:
            learning_rate = st.number_input("📈 學習率", min_value=0.0, value=2e-5, format="%.5f")
            max_length = st.number_input("📏 最大序列長度", min_value=32, value=256)
    
    # 主要内容区域
    st.subheader("📥 準備訓練數據")
    
    # 数据集选择方式
    dataset_source = st.radio(
        "💿 選擇數據集來源",
        ["📁 從資料夾選擇", "📤 上傳新檔案"]
    )
    
    dataset = None
    dataset_path = None
    
    if dataset_source == "📁 從資料夾選擇":
        # 只顯示 qa_dataset.jsonl
        json_files = ["temp_datasets/qa_dataset.jsonl"]
        
        if json_files:
            selected_file = st.selectbox(
                "選擇數據集文件",
                json_files,
                format_func=lambda x: f"📄 {os.path.basename(x)} ({x})"
            )
            
            if selected_file:
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        train_data = [json.loads(line.strip()) for line in lines if line.strip()]
                        dataset_path = selected_file
                        dataset = {"train": train_data}
                        st.success(f"✅ 成功載入數據集：{selected_file}")
                except Exception as e:
                    st.error(f"❌ 載入數據集時發生錯誤：{str(e)}")
        else:
            st.warning("⚠️ 找不到 qa_dataset.jsonl 文件")
    
    else:
        uploaded_file = st.file_uploader("上傳訓練數據集", type=['jsonl'])
        if uploaded_file is not None:
            try:
                temp_dir = "temp_datasets"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                dataset_path = temp_path
                with open(temp_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    train_data = [json.loads(line.strip()) for line in lines if line.strip()]
                    dataset = {"train": train_data}
                
                st.success("✅ 數據集上傳並處理成功！")
            except Exception as e:
                st.error(f"❌ 載入數據集時發生錯誤：{str(e)}")
    
    # 显示数据集信息
    if dataset is not None:
        st.subheader("📊 數據集預覽")
        st.info(f"📈 訓練樣本數量: {len(dataset.get('train', []))}")
        
        with st.expander("👀 查看數據樣本"):
            st.json(dataset['train'][:2])
    
    # 训练部分
    st.subheader("🚀 開始訓練")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 檢查 CUDA 環境", help="點擊以獲取詳細的 CUDA 環境信息"):
            with st.expander("💻 CUDA 環境診斷", expanded=True):
                st.write("=== CUDA 環境診斷 ===")
                st.write(f"🔧 PyTorch 版本: {torch.__version__}")
                st.write(f"✨ CUDA 是否可用: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    st.write(f"💪 GPU 數量: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        st.write(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
                    st.write(f"📌 當前 GPU: {torch.cuda.current_device()}")
                    st.write(f"📊 GPU 記憶體使用量: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    st.write(f"💾 GPU 記憶體快取: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    with col2:
        if st.button("🚀 開始 Fine-tuning", disabled=dataset is None):
            if not torch.cuda.is_available():
                st.warning("⚠️ 警告：未檢測到 GPU，訓練可能會很慢")
            
            output_dir = f"fine_tuned_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                from train import train
                status_text.text("🔄 正在初始化訓練...")
                trainer = train(
                    model_name=model_name,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    learning_rate=learning_rate,
                    max_length=max_length
                )
                
                st.success(f"✅ 訓練完成！模型已保存到 {output_dir}")
                
                if trainer.state.log_history:
                    train_loss = [log.get('loss', 0) for log in trainer.state.log_history if 'loss' in log]
                    eval_loss = [log.get('eval_loss', 0) for log in trainer.state.log_history if 'eval_loss' in log]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=train_loss, name="訓練損失"))
                    if eval_loss:
                        fig.add_trace(go.Scatter(y=eval_loss, name="驗證損失"))
                    fig.update_layout(title="📈 訓練過程", xaxis_title="步驟", yaxis_title="損失")
                    st.plotly_chart(fig)
                    
            except Exception as e:
                st.error(f"❌ 訓練過程中發生錯誤：{str(e)}")
                st.exception(e)

with tab2:
    st.header("🔮 模型推理")
    
    # 模型选择
    model_folders = []
    for folder in os.listdir():
        if folder.startswith('fine_tuned_model_'):
            for checkpoint in os.listdir(folder):
                if checkpoint.startswith('checkpoint-'):
                    checkpoint_path = os.path.join(folder, checkpoint)
                    if os.path.exists(os.path.join(checkpoint_path, 'config.json')):
                        model_folders.append(checkpoint_path)
    
    selected_model = st.selectbox("🤖 选择模型", model_folders)
    
    # 生成参数设置
    st.subheader("⚙️ 生成参数设置")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("🌡️ Temperature", 0.0, 2.0, 0.6, 0.1, 
            help="控制生成文本的随机性。值越高，生成的文本越随机；值越低，生成的文本越确定。")
        max_new_tokens = st.slider("📏 Maximum New Tokens", 1, 2000, 100, 
            help="生成文本的最大长度。")
    with col2:
        top_p = st.slider("📊 Top P", 0.0, 1.0, 0.9, 0.1,
            help="控制生成文本的多样性。值越高，生成的文本越多样；值越低，生成的文本越保守。")
        repetition_penalty = st.slider("🔄 Repetition Penalty", 1.0, 2.0, 1.9, 0.1,
            help="控制重复惩罚程度。值越高，越不容易重复；值越低，越容易重复。")
    
    # 问题输入
    question = st.text_area("💭 输入问题", height=100)
    
    # 创建一个容器来存放回答和签名
    response_container = st.container()
    
    if st.button("🤖 生成回答"):
        if not selected_model:
            st.error("⚠️ 请选择一个模型")
        elif not question:
            st.error("⚠️ 请输入问题")
        else:
            try:
                with st.spinner("⌛ 正在加载模型..."):
                    model, tokenizer = load_model(selected_model)
                    
                with st.spinner("🤔 正在生成回答..."):
                    response = generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        question=question,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty
                    )
                    with response_container:
                        st.markdown("### ✨ 回答")
                        st.write(response)
                        st.markdown('<p class="company-signature">🏢 The Pocket Company by Accucrazy 肖準</p>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ 生成回答时发生错误: {str(e)}")
                st.exception(e)

# 显示使用说明
with st.expander("📖 使用說明"):
    st.markdown("""
    ### 🔄 模型訓練
    1. **選擇數據集**：
        - 📁 從資料夾選擇：選擇已存在的 JSON/JSONL 文件
        - 📤 上傳新檔案：上傳新的數據集文件
    
    2. **配置訓練參數**：
        - ⚙️ 在側邊欄中設置訓練參數
        - 🎮 根據您的 GPU 記憶體大小調整 batch size
    
    3. **開始訓練**：
        - 🚀 點擊"開始 Fine-tuning"按鈕
        - ⏳ 等待訓練完成
    
    ### 🤖 模型推理
    1. **選擇模型**：
        - 📁 從已訓練的模型中選擇
    
    2. **輸入問題**：
        - ✍️ 在文本框中輸入您的問題
    
    3. **生成回答**：
        - 🤖 點擊"生成回答"按鈕
        - 📝 查看模型的回答
    
    ### ⚠️ 注意事項
    - 確保您有足夠的 GPU 記憶體
    - 如果遇到記憶體不足，可以：
        - 📉 減少 batch size
        - 📉 減少最大序列長度
        - 📉 選擇較小的模型版本
    """) 