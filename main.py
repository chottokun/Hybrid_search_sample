from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pickle
import os
from typing import List, Callable # Callable を追加
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_PERSIST_DIR_DEFAULT = "./chroma_data"
BM25_STATE_PATH_DEFAULT = "bm25_state.pkl" # BM25リトリーバーの状態を保存するファイル
DEFAULT_BM25_K = 4 # BM25Retrieverのデフォルトk値

# 重み調整戦略の型定義
WeightAdjustmentStrategy = Callable[[str], List[float]]

class HybridSearchSystem:
    def __init__(self, 
                 embedding_model, 
                 chroma_persist_dir: str = CHROMA_PERSIST_DIR_DEFAULT, 
                 bm25_state_path: str = BM25_STATE_PATH_DEFAULT,
                 weight_adjustment_strategy: WeightAdjustmentStrategy = None):
        self.chroma_persist_dir = chroma_persist_dir
        self.bm25_state_path = bm25_state_path
        self.vector_store = self._init_chroma(embedding_model, self.chroma_persist_dir)
        self.bm25_retriever = self._load_or_initialize_bm25()

        if weight_adjustment_strategy is None:
            self.weight_adjustment_strategy = HybridSearchSystem.default_word_count_strategy
        else:
            self.weight_adjustment_strategy = weight_adjustment_strategy

    def _init_chroma(self, embedding_model, persist_dir):
        return Chroma(
            collection_name="hybrid_search", # コレクション名は固定または設定可能に
            embedding_function=embedding_model,
            persist_directory=persist_dir
        )

    def _initialize_bm25_from_chroma(self):
        """ChromaDBから全ドキュメントを取得してBM25Retrieverを初期化し、状態を保存する"""
        batch_size = 1000
        all_docs = []
        # vector_store._collection.count() が 0 の場合、ループに入らない
        if self.vector_store._collection.count() == 0:
            return None # ドキュメントがなければBM25リトリーバーは作れない

        for offset in range(0, self.vector_store._collection.count(), batch_size):
            batch = self.vector_store._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"] # "ids" も含めると良いかもしれないが、ここでは不要
            )
            # batch["documents"] が None の場合があり得る (ChromaDBのバージョンや状態による)
            # ドキュメントとメタデータのペアを安全に処理
            docs_batch = batch.get("documents")
            metadatas_batch = batch.get("metadatas")

            if docs_batch is None: # ドキュメントがない場合はスキップ
                continue

            for i, doc_content in enumerate(docs_batch):
                if doc_content is None: # 個別のドキュメント内容がNoneの場合もスキップ
                    continue
                # メタデータが存在しない、またはリストの長さが合わない場合のフォールバック
                metadata = {}
                if metadatas_batch and i < len(metadatas_batch) and metadatas_batch[i] is not None:
                    metadata = metadatas_batch[i]
                
                all_docs.append(Document(page_content=doc_content, metadata=metadata))

        if not all_docs:
            return None
        
        retriever = BM25Retriever.from_documents(all_docs)
        retriever.k = DEFAULT_BM25_K # kのデフォルト値を設定
        self._save_bm25_state(retriever)
        return retriever

    def _save_bm25_state(self, retriever):
        if not retriever:
            # BM25リトリーバーがない場合は、既存のstateファイルを削除試行
            if os.path.exists(self.bm25_state_path):
                try:
                    os.remove(self.bm25_state_path)
                except OSError:
                    # logging.warning(f"Failed to remove existing BM25 state file: {self.bm25_state_path}")
                    pass # 削除失敗は許容
            return

        with open(self.bm25_state_path, "wb") as f:
            # BM25Retrieverのdocs (List[Document]) と vectorizer を保存
            # kも保存しておくと復元時に便利
            pickle.dump({
                "docs": retriever.docs, # List[Document]
                "vectorizer": retriever.vectorizer,
                "k": retriever.k
            }, f)

    def _load_bm25_state(self):
        if not os.path.exists(self.bm25_state_path):
            return None
        with open(self.bm25_state_path, "rb") as f:
            state = pickle.load(f)
            # BM25Retrieverのコンストラクタを使って復元
            retriever = BM25Retriever(docs=state["docs"], vectorizer=state["vectorizer"])
            retriever.k = state.get("k", DEFAULT_BM25_K) # 保存されていなければデフォルトk
            return retriever

    def _load_or_initialize_bm25(self):
        retriever = self._load_bm25_state()
        if retriever:
            return retriever
        if self.vector_store._collection.count() > 0:
            return self._initialize_bm25_from_chroma()
        return None

    def _get_ensemble_retriever(self, k: int, weights: List[float]):
        """
        EnsembleRetrieverのインスタンスを生成または取得します。
        kとweightsに基づいてリトリーバーを構成します。
        """
        ensemble_retrievers = []
        actual_weights = []

        if self.bm25_retriever:
            self.bm25_retriever.k = k
            ensemble_retrievers.append(self.bm25_retriever)
            actual_weights.append(weights[0])

        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        ensemble_retrievers.append(vector_retriever)
        
        if self.bm25_retriever:
            actual_weights.append(weights[1])
        else:
            # BM25がない場合、Vectorリトリーバーのみ。重みは1.0とする。
            actual_weights.append(1.0)

        if not ensemble_retrievers:
            return None # 検索可能なリトリーバーがない
        
        # weightsの長さとretrieversの長さが一致していることを確認
        if len(actual_weights) != len(ensemble_retrievers):
            # この状況は通常発生しないはずだが、フォールバック
            # logging.error("Mismatch between number of retrievers and weights.")
            if len(ensemble_retrievers) == 1: # リトリーバーが1つなら重みは[1.0]
                actual_weights = [1.0]
            else: # 不明な場合は均等割り付け（ただしEnsembleRetrieverはエラーになる可能性）
                actual_weights = [1.0/len(ensemble_retrievers)] * len(ensemble_retrievers)

        return EnsembleRetriever(
            retrievers=ensemble_retrievers,
            weights=actual_weights
        )

    def search(self, query: str, k: int = 5):
        """
        同期ハイブリッド検索を実行します。

        Args:
            query (str): 検索クエリ。
            k (int): 各リトリーバーから取得する最大ドキュメント数。
                     EnsembleRetrieverはこれらの結果をマージしてランク付けします。
            weights (List[float]): リトリーバーの重み。[bm25_weight, vector_weight]。
                                   bm25_retrieverが存在しない場合、bm25_weightは無視されます。
                                   このメソッド内部で dynamic_weight_adjuster により決定されます。
        """
        weights = self.dynamic_weight_adjuster(query)
        ensemble_retriever = self._get_ensemble_retriever(k=k, weights=weights)
        if not ensemble_retriever:
            return []
        return ensemble_retriever.invoke(query)

    async def asearch(self, query: str, k: int = 5):
        """
        非同期ハイブリッド検索を実行します。
        """
        weights = self.dynamic_weight_adjuster(query)
        ensemble_retriever = self._get_ensemble_retriever(k=k, weights=weights)
        if not ensemble_retriever:
            return []
        return await ensemble_retriever.ainvoke(query)

    @staticmethod
    def default_word_count_strategy(query: str) -> List[float]:
        """デフォルトの単語数に基づく重み調整戦略。"""
        # クエリの単語数に基づいて重みを動的に調整
        # 短いクエリ（キーワード検索に近い）場合はBM25の重みを上げる
        # 長いクエリ（意味検索に近い）場合はベクトル検索の重みを上げる
        if len(query.split()) < 3: # 例: 3単語未満ならキーワード検索寄り
            return [0.7, 0.3]  # [bm25_weight, vector_weight]
        return [0.3, 0.7]      # [bm25_weight, vector_weight]
    
    @staticmethod
    def question_based_strategy(query: str) -> List[float]:
        """疑問詞の有無と単語数に基づく重み調整戦略。"""
        question_words = [
            "what", "who", "where", "when", "why", "how", "which", "whose", "whom",
            "何", "誰", "どこ", "いつ", "なぜ", "どうして", "どのように", "どの", "どちら", "どなた"
        ]
        query_lower = query.lower()
        is_question = any(q_word in query_lower for q_word in question_words)

        if is_question:
            return [0.2, 0.8]  # 質問形式ならVector重視
        elif len(query.split()) < 3:
            return [0.7, 0.3]  # 短い非質問クエリはBM25重視
        else:
            return [0.5, 0.5]  # それ以外はバランス型

    def add_documents(self, docs: List[Document]):
        """
        ドキュメントをChromaベクトルストアとBM25リトリーバーに追加（または更新）します。
        BM25リトリーバーの状態は永続化されます。
        """
        if not docs:
            return
        
        # Chromaに追加
        self.vector_store.add_documents(docs)

        # BM25リトリーバーを更新
        if self.bm25_retriever is None:
            # BM25が未初期化の場合、Chromaの全データから構築 (渡されたdocsもChroma経由で含まれる)
            self.bm25_retriever = self._initialize_bm25_from_chroma()
        else:
            self.bm25_retriever.add_documents(docs) # 既存のBM25リトリーバーにドキュメントを追加
            self._save_bm25_state(self.bm25_retriever) # 更新後に状態を保存

    def dynamic_weight_adjuster(self, query: str) -> List[float]:
        """設定された戦略に基づいて動的に重みを調整します。"""
        return self.weight_adjustment_strategy(query)

    def close(self):
        """
        ChromaDBの接続を明示的に閉じる
        """
        if hasattr(self.vector_store, '_client') and self.vector_store._client is not None:
            # Note: PersistentClient in chromadb version used might not have a close() method,
            # or it might be managed differently. Langchain's Chroma class handles client lifecycle.
            # ChromaDBのPersistentClientには明示的なclose()メソッドがない場合がある。
            # LangchainのChromaクラスが内部でどのようにクライアントを扱っているかによる。
            # 通常、アプリケーション終了時に自動的にリソースが解放される。
            # もしエラーが出るようならコメントアウト。
            # self.vector_store._client. κάποια_μέθοδος_κλεισίματος() # 例
            print("ChromaDB接続のクローズ処理を試みました（実際のクローズはChromaライブラリの実装依存）。")

    def reindex_bm25_from_chroma(self):
        """ChromaDBの全データからBM25インデックスを強制的に再構築します。"""
        self.bm25_retriever = self._initialize_bm25_from_chroma()
        if self.bm25_retriever:
            print("BM25インデックスの再構築が完了し、保存されました。")
        else:
            print("ドキュメントがないため、BM25インデックスは構築されませんでした。BM25 stateファイルも削除試行されます。")
            self._save_bm25_state(None) # BM25リトリーバーがない場合はstateファイルを削除

    def delete_all_chroma_data(self):
        """ChromaDBの全てのコレクションを削除します。BM25の永続化ファイルも削除します。"""
        try:
            collections = self.vector_store._client.list_collections()
            for collection in collections:
                self.vector_store._client.delete_collection(collection.name)
            print(f"ChromaDBの全{len(collections)}コレクションを削除しました。")
            self._save_bm25_state(None) # BM25リトリーバーがない状態として保存（実質stateファイル削除）
            self.bm25_retriever = None # メモリ上のリトリーバーもクリア
            print(f"BM25永続化ファイル ({self.bm25_state_path}) も削除（または削除試行）しました。")
        except Exception as e:
            print(f"全データ削除中にエラーが発生しました: {e}")


# 日本語BM25対応（形態素解析）
# この部分は変更なし
try:
    from sudachipy import tokenizer
    from sudachipy import dictionary

    sudachi_tokenizer = dictionary.Dictionary().create()
    sudachi_mode = tokenizer.Tokenizer.SplitMode.C

    # JapaneseBM25Retrieverクラスは現状使われていないが、定義は残す
    class JapaneseBM25Retriever(BM25Retriever): # type: ignore
        def _tokenize(self, text: str) -> List[str]:
            # sudachipyのトークナイザモードを確認。通常はSplitMode.A, B, Cがある。
            # ここではCを使用。
            return [m.surface() for m in sudachi_tokenizer.tokenize(text, sudachi_mode)]
except ImportError:
    JapaneseBM25Retriever = None # type: ignore


def main_cli(): # streamlitと区別するためmainから変更
    print("Hybrid Search CLI")
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # Chromaデータディレクトリを指定
    chroma_data_dir = os.path.join(os.path.dirname(__file__), "chroma_data_cli")
    os.makedirs(chroma_data_dir, exist_ok=True)
    
    # 使用する戦略を選択
    # ここでは、デフォルトの挙動（コンストラクタ内で default_word_count_strategy が設定される）を利用する。
    # system = HybridSearchSystem(embedding_model, chroma_data_dir) 
    
    # 別の戦略を試す場合:
    # system = HybridSearchSystem(embedding_model, chroma_data_dir, weight_adjustment_strategy=HybridSearchSystem.question_based_strategy)
    system = HybridSearchSystem(embedding_model, chroma_data_dir) # デフォルト戦略を使用
    print(f"使用中の重み調整戦略: {system.weight_adjustment_strategy.__name__}")
    print(f"ChromaDBデータディレクトリ: {system.chroma_persist_dir}")
    print(f"BM25状態ファイル: {system.bm25_state_path}")
    if system.bm25_retriever:
        print(f"BM25リトリーバーがロード/初期化されました。ドキュメント数: {len(system.bm25_retriever.docs)}")
    else:
        print("BM25リトリーバーは利用できません（データなし）。")


    while True:
        cmd = input("add/search/reindex_bm25/delete_all/exit > ").strip().lower()
        if cmd == "add":
            text = input("追加するドキュメントを入力: ")
            if not text.strip():
                print("空のドキュメントは追加できません。")
                continue
            # 手動追加時もDocumentオブジェクトを作成
            doc_to_add = [Document(page_content=text, metadata={"source": "manual_cli_input"})]
            system.add_documents(doc_to_add)
            print("ドキュメントをChromaとBM25に追加（または更新）し、BM25の状態を保存しました。")
        elif cmd == "search":
            query = input("検索クエリを入力: ")
            if not query.strip():
                print("検索クエリを入力してください。")
                continue
            weights = system.dynamic_weight_adjuster(query) # 参考: searchメソッド内部で呼ばれる
            print(f"使用する重み (BM25, Vector): {weights}")
            results = system.search(query, k=5) # weightsはsearch内部で決定
            print("\n検索結果:")
            if results:
                for i, doc in enumerate(results, 1):
                    print(f"  {i}. Content: {doc.page_content}")
                    if doc.metadata:
                        print(f"     Metadata: {doc.metadata}")
            else:
                print("  該当する結果はありませんでした。")
            print("-" * 20)
        elif cmd == "reindex_bm25":
            print("BM25インデックスをChromaDBの全データから再構築します...")
            system.reindex_bm25_from_chroma()
        elif cmd == "delete_all":
            confirm = input("本当にChromaDBの全データとBM25 stateを削除しますか？ (yes/no): ").strip().lower()
            if confirm == "yes":
                system.delete_all_chroma_data()
                print("全データの削除処理を実行しました。")
        elif cmd == "exit":
            print("終了します。")
            break
        else:
            print("コマンドが不正です。(add, search, reindex_bm25, delete_all, exit)")


if __name__ == "__main__":
    # Streamlitから呼び出される場合は `main_cli` は実行されない
    # このファイルが直接実行された場合のみ `main_cli` を実行
    # (例: python main.py)
    main_cli()
