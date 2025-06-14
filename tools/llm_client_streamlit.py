import streamlit as st
from llm_client import LLMClient

# ダミー設定クラス（本番は適宜差し替え）
class MockConfiguration:
    LLM_PROVIDER = "placeholder_llm"
    LLM_MODEL = "test_model"
    LLM_MAX_TOKENS = 256
    OPENAI_API_KEY = None
    LOG_LEVEL = "DEBUG"
    LLM_TEMPERATURE = 0.7

# LLMClientの初期化
def get_llm_client():
    config = MockConfiguration()
    return LLMClient(config=config)

st.title("LLMClient 簡易テストUI (Streamlit)")

prompt = st.text_area("プロンプトを入力してください", "AIとは何ですか？")

if st.button("送信"):
    llm_client = get_llm_client()
    with st.spinner("LLM応答を生成中..."):
        try:
            response = llm_client.generate_text(prompt)
            st.success("LLM応答:")
            st.write(response)
        except Exception as e:
            st.error(f"エラー: {e}")
