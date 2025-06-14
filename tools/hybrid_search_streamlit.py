import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import os
import shutil  # 追加
import importlib # 追加 # mainモジュールの動的リロード用
from typing import Callable, List # 型ヒント用
import hashlib # ドキュメントのハッシュ化用
import datetime # タイムスタンプ用

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main # mainモジュールをインポート
importlib.reload(main) # mainモジュールを再ロード
from main import HybridSearchSystem, WeightAdjustmentStrategy # HybridSearchSystemと型エイリアスをインポート
from tools.pdf_utils import extract_texts_from_pdfs
from langchain_core.documents import Document # Documentクラスをインポート

# 利用可能な戦略の定義
AVAILABLE_STRATEGIES: dict[str, WeightAdjustmentStrategy] = {
    "単語数ベース（デフォルト）": HybridSearchSystem.default_word_count_strategy,
    "疑問詞ベース": HybridSearchSystem.question_based_strategy,
}

# 検索システム初期化（リソースキャッシュ）
@st.cache_resource
def get_system_instance(selected_strategy_name: str): # Changed parameter to accept name (string)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # main.pyのデフォルトパスを使用
    chroma_persist_dir = main.CHROMA_PERSIST_DIR_DEFAULT
    bm25_state_path = main.BM25_STATE_PATH_DEFAULT

    # Retrieve the actual function using the name
    actual_strategy_func = AVAILABLE_STRATEGIES[selected_strategy_name]

    return HybridSearchSystem(
        embedding_model,
        chroma_persist_dir=chroma_persist_dir,
        bm25_state_path=bm25_state_path,
        weight_adjustment_strategy=actual_strategy_func # Pass the retrieved function object
    )

# --- セッションステートの初期化とUI ---
st.title("ハイブリッド検索テスト (Streamlit)")

# 戦略選択UI
if 'selected_strategy_name' not in st.session_state:
    st.session_state.selected_strategy_name = list(AVAILABLE_STRATEGIES.keys())[0] # 初期戦略

selected_strategy_name = st.sidebar.selectbox(
    "重み調整戦略を選択:",
    options=list(AVAILABLE_STRATEGIES.keys()),
    key='selected_strategy_name_selector' # st.session_state.selected_strategy_name_selector に値が入る
)
# selectboxの変更をselected_strategy_nameに反映（ページ再実行時に反映される）
st.session_state.selected_strategy_name = selected_strategy_name
current_strategy_func = AVAILABLE_STRATEGIES[st.session_state.selected_strategy_name]

# セッションステートでシステムを管理
if 'system' not in st.session_state or \
   st.session_state.system.weight_adjustment_strategy != current_strategy_func:
    # Call with the strategy NAME (string), which is hashable
    st.session_state.system = get_system_instance(st.session_state.selected_strategy_name)

system = st.session_state.system
st.sidebar.info(f"現在の戦略: {st.session_state.selected_strategy_name}")

# --- いいねフィードバック用のセッションステート初期化 ---
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

st.header("ドキュメント追加")
doc_text = st.text_area("追加するドキュメントを入力")
if st.button("ドキュメント追加"):
    if doc_text.strip():
        try:
            # 手動追加時もDocumentオブジェクトを作成
            doc_to_add = [Document(page_content=doc_text, metadata={"source": "manual_input"})]
            # system.add_documents_to_chroma(doc_to_add) # 古い呼び出し
            # system.add_documents_to_bm25(doc_to_add)   # 古い呼び出し
            system.add_documents(doc_to_add) # 統合されたメソッド呼び出し
            st.success("ドキュメントを追加しました。")
        except Exception as e:
            st.error(f"追加エラー: {e}")
    else:
        st.warning("空のドキュメントは追加できません。")

st.header("検索")
query = st.text_input("検索クエリを入力してください")

if st.button("検索"):
    with st.spinner("検索中..."):
        try:
            # weights = system.dynamic_weight_adjuster(query) # searchメソッド内部で処理される
            results = system.search(query, k=5) # weights引数は不要
            if results:
                st.success(f"検索結果: {len(results)} 件")
                for i, doc in enumerate(results, 1):
                    st.markdown(f"---")
                    doc_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
                    
                    col1, col2 = st.columns([0.9, 0.1])
                    with col1:
                        st.markdown(f"**No.{i}**")
                        st.markdown(f"**内容:**\n{doc.page_content}")
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.markdown(f"**メタデータ:** {doc.metadata}")
                    with col2:
                        if st.button("👍", key=f"like_{query}_{doc_hash}_{i}"):
                            feedback_entry = {
                                "query": query,
                                "document_hash": doc_hash,
                                "document_content_preview": doc.page_content[:100] + "...", # 先頭100文字
                                "liked_at": datetime.datetime.now().isoformat(),
                                "strategy_used": st.session_state.selected_strategy_name,
                                "rank": i
                            }
                            st.session_state.feedback_data.append(feedback_entry)
                            st.toast("フィードバックを記録しました！", icon="👍")
            else:
                st.info("該当する検索結果がありません。")
        except Exception as e:
            st.error(f"エラー: {e}")

st.header("PDF一括インポート（静的update）")
# PDFチャンクサイズとオーバーラップの入力
chunk_size_pdf = st.number_input("チャンクサイズ (PDF)", min_value=100, max_value=2000, value=500, step=50, help="PDFを分割する際の各チャンクの最大文字数。")
chunk_overlap_pdf = st.number_input("チャンクオーバーラップ (PDF)", min_value=0, max_value=500, value=50, step=10, help="チャンク間の重複文字数。")

if st.button("PDFディレクトリから全件再構築"):
    with st.spinner("PDFからテキスト抽出・インデックス再構築中..."):
        try:
            pdf_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pdf"))
            st.write(f"PDFディレクトリ: {pdf_dir}")
            
            # main.pyからデフォルトパスを取得
            chroma_persist_dir = main.CHROMA_PERSIST_DIR_DEFAULT
            bm25_state_path = main.BM25_STATE_PATH_DEFAULT
            st.write(f"Chroma永続化ディレクトリ: {os.path.abspath(chroma_persist_dir)}")
            st.write(f"BM25状態ファイルパス: {os.path.abspath(bm25_state_path)}")
            
            # 既存のChromaDBインスタンスを破棄し、ディレクトリを削除
            if 'system' in st.session_state and st.session_state.system is not None:
                st.session_state.system.close() # 明示的にChromaDB接続を閉じる
                del st.session_state.system # 既存のシステムインスタンスを削除
            if os.path.exists(chroma_persist_dir):
                st.write(f"既存のChromaディレクトリを削除中: {chroma_persist_dir}")
                shutil.rmtree(chroma_persist_dir)
                st.write("既存のChromaディレクトリを削除しました。")
            if os.path.exists(bm25_state_path):
                st.write(f"既存のBM25状態ファイルを削除中: {bm25_state_path}")
                os.remove(bm25_state_path)
                st.write("既存のBM25状態ファイルを削除しました。")
            
            os.makedirs(chroma_persist_dir, exist_ok=True)
            # st.write(f"作成直後のchroma_data中身: {os.listdir(chroma_persist_dir)}") # ディレクトリが空なので表示は省略可
            
            # 新しいシステムインスタンスを作成
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            current_strategy_func_for_rebuild = AVAILABLE_STRATEGIES[st.session_state.selected_strategy_name]
            new_system = HybridSearchSystem(
                embedding_model,
                chroma_persist_dir=chroma_persist_dir,
                bm25_state_path=bm25_state_path,
                weight_adjustment_strategy=current_strategy_func_for_rebuild
            )
            st.session_state.system = new_system # セッションステートに新しいシステムを保存
            system = new_system # 現在のシステム変数も更新
            
            # st.write(f"Chroma初期化後のchroma_data中身: {os.listdir(chroma_persist_dir)}")
            # チャンクサイズとオーバーラップを指定してPDFからDocumentを抽出
            pdf_documents = extract_texts_from_pdfs(pdf_dir, chunk_size=chunk_size_pdf, chunk_overlap=chunk_overlap_pdf)
            st.write(f"抽出されたドキュメントチャンク数: {len(pdf_documents)}")
            if not pdf_documents:
                st.warning("PDFからテキストが抽出できませんでした。")
            else:
                system.add_documents(pdf_documents) # 統合されたメソッド呼び出し
                # st.write(f"add_documents後のchroma_data中身: {os.listdir(chroma_persist_dir)}")
                st.success(f"{len(pdf_documents)}件のドキュメントチャンクでインデックスを再構築しました。")
                st.info("インデックスの再構築が完了しました。") # リロードは不要なはず
        except Exception as e:
            st.error(f"PDF一括インポート失敗: {e}")

# 全件削除ボタン
if st.button("Chroma全件削除"):
    try:
        system.delete_all_chroma_data() # 統合された削除メソッド
        st.success("ChromaDBの全データとBM25 stateを削除しました。")
        # 必要であればシステムインスタンスをリセット
        # st.cache_resource.clear()
        # del st.session_state.system
        # st.experimental_rerun() # ページを再実行してシステムを再初期化
    except Exception as e:
        st.error(f"全データ削除エラー: {e}")

# キャッシュクリアボタン
if st.sidebar.button("キャッシュクリア＆リロード"):
    st.cache_resource.clear()
    st.session_state.clear()
    st.success("キャッシュとセッションをクリアしました。ページをリロードしてください。")
    st.experimental_rerun() # Streamlit 1.19.0以降
    # st.stop() # 古いバージョン用
