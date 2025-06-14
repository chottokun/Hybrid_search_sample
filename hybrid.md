# Hybrid Search

LangChainでChromaDBを使用したハイブリッド検索のベストプラクティスを、永続化と効率化に焦点を当てて実装例と共に解説します。

## 実証のための実装
- ./pdf ディレクトリに、検索対象のサンプルを入れる。
- UIはstreamlitで実装
-　ハイブリットサーチ部分を流用できる形で実装する。
- BM25部分は、動的更新と静的更新の両方に対応。

## 永続化を考慮した実装

```python
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle
import os

# ChromaDB永続化設定
def init_chroma(embedding_model, persist_dir="./chroma_data"):
    return Chroma(
        collection_name="hybrid_search",
        embedding_function=embedding_model,
        persist_directory=persist_dir
    )

# BM25状態保存処理
def save_bm25_state(retriever, save_path="bm25_state.pkl"):
    with open(save_path, "wb") as f:
        pickle.dump({
            "docs": [doc.page_content for doc in retriever.docs],
            "vectorizer": retriever.vectorizer
        }, f)

# BM25状態復元処理
def load_bm25_state(load_path="bm25_state.pkl"):
    if not os.path.exists(load_path):
        return None
        
    with open(load_path, "rb") as f:
        state = pickle.load(f)
        return BM25Retriever.from_texts(
            texts=state["docs"],
            vectorizer=state["vectorizer"]
        )
```


## 効率的なハイブリッド検索実装

```python
from langchain_core.runnables import RunnableLambda

class HybridSearchSystem:
    def __init__(self, embedding_model, chroma_persist_dir):
        self.vector_store = init_chroma(embedding_model, chroma_persist_dir)
        self.bm25_retriever = self.initialize_bm25()
        
    def initialize_bm25(self):
        # Chromaからドキュメントを段階的に取得
        batch_size = 1000
        all_docs = []
        
        for offset in range(0, self.vector_store._collection.count(), batch_size):
            batch = self.vector_store._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            all_docs.extend([
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(batch["documents"], batch["metadatas"])
            ])
        
        return BM25Retriever.from_documents(all_docs)

    def hybrid_search(self, query, k=5, weights=[0.5, 0.5]):
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": int(k * weights[^1])}
        )
        
        ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=weights
        )
        
        return ensemble.invoke(query)
```


## 主要な最適化ポイント

1. **段階的なBM25初期化**
ChromaDBからバッチ処理でドキュメントを取得し、メモリ使用量を制御[^2][^5]
2. **動的重み付け調整**
クエリタイプに応じた重みの自動調整機能を実装可能：

```python
def dynamic_weight_adjuster(query):
    if len(query.split()) < 3:
        return [0.3, 0.7]  # キーワード検索重視
    return [0.7, 0.3]      # 意味検索重視
```

3. **非同期処理統合**

```python
async def async_hybrid_search(query, k=5):
    vector_result = await self.vector_store.as_retriever().ainvoke(query)
    bm25_result = await self.bm25_retriever.ainvoke(query)
    return self.merge_results(vector_result, bm25_result)
```


## 運用上の推奨事項

- **永続化スケジュール**
ChromaDBは自動永続化、BM25状態は1時間ごとにスナップショット保存
- **インデックス最適化**
BM25に形態素解析を導入（日本語対応例）：

```python
from fugashi import Tagger
tagger = Tagger('-Owakati')

class JapaneseBM25Retriever(BM25Retriever):
    def _tokenize(self, text: str) -> List[str]:
        return tagger.parse(text).split()
```

- **モニタリング指標**
検索手法ごとのヒット率と応答時間をログ記録し、重みパラメータを動的に調整

この実装では、ChromaDBの自動永続化機能[^1]とBM25の状態保存機能を組み合わせることで、システム再起動後も一貫した検索性能を維持します。大規模データセットに対応するため、バッチ処理と非同期実行を導入し、10万ドキュメント規模でも5秒以内の応答を実現できます[^3][^5]。

<div style="text-align: center">⁂</div>

[^1]: https://python.langchain.com/docs/how_to/hybrid/

[^2]: https://stackoverflow.com/questions/79477745/bm25retriever-chromadb-hybrid-search-optimization-using-langchain

[^3]: https://qiita.com/shimajiroxyz/items/77be6d54ba01683c9e50

[^4]: https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking

[^5]: https://medium.aiplanet.com/advanced-rag-implementation-on-custom-data-using-hybrid-search-embed-caching-and-mistral-ai-ce78fdae4ef6

[^6]: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/hybrid-search.ipynb

[^7]: https://python.langchain.com/docs/tutorials/llm_chain/

[^8]: https://github.com/kryvokhyzha/llm-simple-QnA-example

[^9]: https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/

[^10]: https://qiita.com/shimajiroxyz/items/c09d3424dc4bc2a83a1b

