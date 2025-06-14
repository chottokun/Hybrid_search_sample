import streamlit as st

st.set_page_config(
    page_title="Hybrid Search System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("ハイブリッド検索システムへようこそ")
st.sidebar.success("上のナビゲーションからページを選択してください。")

st.markdown(
    """
    このアプリケーションは、ハイブリッド検索（ベクトル検索とキーワード検索の組み合わせ）の
    テストと評価を行うためのものです。

    **主な機能:**
    - 🔍 **ハイブリッド検索**: ドキュメントの追加、検索、PDFからの一括インポート。 (pages/1_🔍_Hybrid_Search.py)
    - 📊 **戦略評価**: 定義した評価セットに基づいて、異なる検索戦略の有効性を比較。 (pages/2_📊_Strategy_Evaluation.py)
    - 👍 **フィードバック分析**: ユーザーからの検索結果への「いいね」フィードバックを収集・分析。 (pages/3_👍_Feedback_Analysis.py)

    左のサイドバーから各機能ページへアクセスしてください。
    """
)