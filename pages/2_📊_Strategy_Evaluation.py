import streamlit as st
import sys
import os
import importlib
from typing import Callable, List # 型ヒント用

# 親ディレクトリをsys.pathに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import main # mainモジュールをインポート
importlib.reload(main) # mainモジュールを再ロード
from main import HybridSearchSystem, WeightAdjustmentStrategy # HybridSearchSystemと型エイリアスをインポート
from langchain_huggingface import HuggingFaceEmbeddings

# 利用可能な戦略の定義 (メインのStreamlitアプリと共通化することも検討)
AVAILABLE_STRATEGIES: dict[str, WeightAdjustmentStrategy] = {
    "単語数ベース（デフォルト）": HybridSearchSystem.default_word_count_strategy,
    "疑問詞ベース": HybridSearchSystem.question_based_strategy,
}

# 検索システム初期化（リソースキャッシュ）
@st.cache_resource
def get_system_instance_for_evaluation(selected_strategy_func: WeightAdjustmentStrategy):
    # この関数はメインアプリの get_system_instance とほぼ同じだが、
    # 評価専用のインスタンスを生成する場合や、設定を少し変えたい場合に分離できる。
    # ここではメインアプリと同じ設定でインスタンスを生成する。
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chroma_persist_dir = main.CHROMA_PERSIST_DIR_DEFAULT
    bm25_state_path = main.BM25_STATE_PATH_DEFAULT
    return HybridSearchSystem(
        embedding_model,
        chroma_persist_dir=chroma_persist_dir,
        bm25_state_path=bm25_state_path,
        weight_adjustment_strategy=selected_strategy_func # 初期戦略として渡す
    )

st.set_page_config(page_title="戦略評価", layout="wide")
st.title("ハイブリッド検索戦略の評価")

if 'evaluation_sets' not in st.session_state:
    st.session_state.evaluation_sets = []

with st.expander("評価セットの定義", expanded=False):
    eval_query = st.text_input("評価用クエリ", key="eval_query_input_page")
    eval_expected_keyword = st.text_input("期待するキーワード", key="eval_keyword_input_page")
    eval_top_n = st.number_input("期待する順位 (Top N)", min_value=1, value=3, key="eval_top_n_input_page")

    if st.button("評価セットに追加", key="add_eval_set_button_page"):
        if eval_query.strip() and eval_expected_keyword.strip():
            st.session_state.evaluation_sets.append({
                "query": eval_query,
                "expected_keyword": eval_expected_keyword,
                "top_n": eval_top_n
            })
            st.success("評価セットに追加しました。")
            st.session_state.eval_query_input_page = ""
            st.session_state.eval_keyword_input_page = ""
            st.session_state.eval_top_n_input_page = 3
        else:
            st.warning("評価用クエリと期待するキーワードを入力してください。")

st.subheader("現在の評価セット")
if not st.session_state.evaluation_sets:
    st.info("評価セットがまだ定義されていません。")
else:
    for i, es in enumerate(st.session_state.evaluation_sets):
        st.markdown(f"- クエリ: `{es['query']}`, キーワード: `{es['expected_keyword']}`, 期待順位: Top {es['top_n']}")

if st.button("全戦略で評価を実行", disabled=not st.session_state.evaluation_sets, key="run_eval_button_page"):
    # 評価時には、選択されている戦略に関わらず、定義済みの全戦略を試す
    # そのため、システムインスタンスは評価ループ内で戦略を切り替えるか、
    # 戦略ごとにインスタンスを生成する（キャッシュが効く）
    evaluation_results = {}
    with st.spinner("評価を実行中..."):
        for strategy_name, strategy_func in AVAILABLE_STRATEGIES.items():
            # 戦略ごとに新しい（またはキャッシュされた）システムインスタンスを取得
            eval_system = get_system_instance_for_evaluation(strategy_func)
            if not eval_system.vector_store._collection.count() > 0 and not eval_system.bm25_retriever :
                 st.error(f"戦略「{strategy_name}」の評価スキップ: データがロードされていません。メインページでデータを追加/インポートしてください。")
                 continue

            strategy_scores = []
            for es_item in st.session_state.evaluation_sets:
                results = eval_system.search(es_item["query"], k=max(10, es_item["top_n"]))
                found_rank = -1
                for rank, doc in enumerate(results[:es_item["top_n"]]):
                    if es_item["expected_keyword"].lower() in doc.page_content.lower():
                        found_rank = rank + 1
                        break
                strategy_scores.append({"query": es_item["query"], "expected_keyword": es_item["expected_keyword"], "rank": found_rank, "target_top_n": es_item["top_n"]})
            evaluation_results[strategy_name] = strategy_scores

    st.subheader("評価結果")
    for strategy_name, scores in evaluation_results.items():
        st.markdown(f"**戦略: {strategy_name}**")
        successful_queries = 0
        if not scores: # データロードエラーなどでscoresが空の場合
            st.warning("この戦略の評価データがありません。")
            continue
        for score in scores:
            if score["rank"] != -1:
                st.markdown(f"  - クエリ「{score['query']}」でキーワード「{score['expected_keyword']}」が **{score['rank']}位** で見つかりました (期待: Top {score['target_top_n']})。")
                successful_queries +=1
            else:
                st.markdown(f"  - クエリ「{score['query']}」でキーワード「{score['expected_keyword']}」が期待順位Top {score['target_top_n']}以内に見つかりませんでした。")
        st.markdown(f"  -> 成功率: {successful_queries / len(scores) * 100:.2f}% ({successful_queries}/{len(scores)})")
        st.markdown("---")

st.sidebar.page_link("pages/1_🔍_Hybrid_Search.py", label="メイン検索ページへ")
st.sidebar.page_link("pages/3_👍_Feedback_Analysis.py", label="フィードバック分析ページへ")