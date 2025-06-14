import streamlit as st
import pandas as pd
from collections import Counter

st.set_page_config(page_title="フィードバック分析", layout="wide")
st.title("👍 検索結果フィードバック分析")

if 'feedback_data' not in st.session_state or not st.session_state.feedback_data:
    st.info("まだフィードバックデータがありません。メイン検索ページで検索結果に「いいね」をしてください。")
    st.stop()

feedback_df = pd.DataFrame(st.session_state.feedback_data)

st.header("全フィードバックデータ")
st.dataframe(feedback_df)

st.header("フィードバックサマリー")

# 1. クエリ別いいね数
st.subheader("クエリ別いいね数ランキング")
query_likes = Counter(feedback_df["query"])
query_likes_df = pd.DataFrame(query_likes.items(), columns=["クエリ", "いいね数"]).sort_values(by="いいね数", ascending=False)
st.dataframe(query_likes_df)

# 2. ドキュメント別いいね数（プレビューで表示）
st.subheader("ドキュメント別いいね数ランキング")
doc_likes_data = []
for _, row in feedback_df.iterrows():
    doc_likes_data.append((row["document_content_preview"], row["document_hash"]))

doc_likes_counter = Counter(doc_likes_data)
doc_likes_list = []
for (preview, doc_hash), count in doc_likes_counter.most_common():
    doc_likes_list.append({"ドキュメントプレビュー": preview, "いいね数": count, "ハッシュ": doc_hash})

if doc_likes_list:
    doc_likes_df = pd.DataFrame(doc_likes_list)
    st.dataframe(doc_likes_df)
else:
    st.info("ドキュメントへの「いいね」がまだありません。")


# 3. 戦略別いいね数
st.subheader("検索戦略別いいね数")
strategy_likes = Counter(feedback_df["strategy_used"])
strategy_likes_df = pd.DataFrame(strategy_likes.items(), columns=["戦略", "いいね数"]).sort_values(by="いいね数", ascending=False)
st.dataframe(strategy_likes_df)

# 4. いいねされた時の検索順位の分布
st.subheader("「いいね」された結果の検索順位分布")
if "rank" in feedback_df.columns:
    rank_counts = Counter(feedback_df["rank"])
    rank_df = pd.DataFrame(rank_counts.items(), columns=["順位", "回数"]).sort_values(by="順位")
    st.bar_chart(rank_df.set_index("順位"))
else:
    st.info("順位データを含むフィードバックがありません。")


st.sidebar.page_link("pages/1_🔍_Hybrid_Search.py", label="メイン検索ページへ")
st.sidebar.page_link("pages/2_📊_Strategy_Evaluation.py", label="戦略評価ページへ")


if st.sidebar.button("フィードバックデータをクリア（セッション内）"):
    st.session_state.feedback_data = []
    st.toast("セッション内のフィードバックデータをクリアしました。")
    st.experimental_rerun()