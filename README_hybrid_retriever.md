# HybridRetriever モジュールの使い方（LangChain RAG対応）

このモジュールは、Chroma（ベクトル検索）とBM25（キーワード検索）の長所を組み合わせた高度なハイブリッド検索機能を提供します。LangChainの標準的な`Retriever`インターフェースとして設計されているため、既存のLangChain RAG (Retrieval Augmented Generation) パイプラインや他のアプリケーションへ容易に組み込むことができます。これにより、セマンティックな関連性とキーワードのマッチング精度を両立させた、より効果的な情報検索を実現します。

## 主な特徴
- **LangChain `BaseRetriever`互換**: 既存のLangChain RAGパイプラインに容易に統合可能です。
- **柔軟な検索戦略**: ベクトル検索とBM25検索の重み付けを調整可能 (`weights`引数)。
- **独立したコンポーネント利用**: 既存のベクトルリトリーバーやBM25リトリーバーと組み合わせて利用できます。
- **結果の自動重複排除**: 複数の検索手法からの結果をマージする際に、重複するドキュメントは自動的に排除されます。

## 1. 構成ファイル
- `tools/hybrid_retriever.py` : HybridRetrieverクラス本体
- `main.py` : `HybridSearchSystem`クラスを含む、Chroma/BM25の初期化やデータ追加のサンプル実装。

## 2. 必要なパッケージ
- langchain
- langchain-core
- chromadb
- langchain-chroma
- langchain-community
- sudachipy（日本語テキストでBM25検索の精度を高めるために推奨、任意）
- sudachidict_core（sudachipyの辞書、任意）

## 3. HybridRetriever の初期化と利用方法

### 3.1. `HybridSearchSystem` を利用した初期化 (推奨される簡単な方法)

`main.py` に含まれる `HybridSearchSystem` は、Chromaデータベースの構築、BM25リトリーバーの準備、およびこれらのコンポーネントの管理を簡略化するヘルパークラスです。
このプロジェクトのサンプルを手軽に試したい場合や、迅速にハイブリッド検索をセットアップしたい場合に便利です。

```python
from tools.hybrid_retriever import HybridRetriever
from main import HybridSearchSystem # main.pyからHybridSearchSystemをインポート
from langchain_huggingface import HuggingFaceEmbeddings

# ベクトル埋め込みモデルの準備
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# HybridSearchSystemを使用してChromaとBM25Retrieverを初期化
# (内部でデータのロード、Chroma DBの構築、BM25Retrieverの準備が行われます)
system = HybridSearchSystem(
    embedding_model=embedding_model,
    chroma_persist_dir="./chroma_data" # Chromaの永続化ディレクトリ
)
# system.add_documents(texts) # 必要に応じてドキュメントを追加

# LangChain Retrieverとして使えるハイブリッドリトリーバーを作成
hybrid_retriever_via_system = HybridRetriever(
    vector_retriever=system.vector_store.as_retriever(search_kwargs={"k": 5}), # ベクトル検索で5件取得
    bm25_retriever=system.bm25_retriever # BM25Retriever (kは内部で設定されているか、別途設定)
)

# BM25Retrieverのk値も設定する場合
if system.bm25_retriever:
    system.bm25_retriever.k = 5 # BM25検索で5件取得

hybrid_retriever_via_system = HybridRetriever(
    vector_retriever=system.vector_store.as_retriever(search_kwargs={"k": 5}),
    bm25_retriever=system.bm25_retriever,
    weights=[0.5, 0.5] # 重み付けの例
)
```

### 3.2. 個別のリトリーバーコンポーネントからの直接初期化

既にLangChainのベクトルリトリーバー（例: Chroma）やBM25リトリーバーのインスタンスをお持ちの場合、または `HybridSearchSystem` を介さずに各コンポーネントをより細かく制御したい場合にこの方法を使用します。これにより、`HybridRetriever` を既存のシステムへ柔軟に組み込むことができます。

```python
from tools.hybrid_retriever import HybridRetriever
from main import HybridSearchSystem
from langchain_huggingface import HuggingFaceEmbeddings

# ベクトル埋め込みモデル
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# HybridSearchSystemでChroma/BM25を初期化
system = HybridSearchSystem(embedding_model, "./chroma_data")

# LangChain Retrieverとして使えるハイブリッドリトリーバーを作成
hybrid_retriever = HybridRetriever(
    vector_retriever=system.vector_store.as_retriever(),
    bm25_retriever=system.bm25_retriever
)
```

## 4. LangChainのRAGチェーンで利用する例
```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=hybrid_retriever,
    return_source_documents=True
)

result = qa_chain({"query": "検索したい内容"})
print(result["result"])
```

## 4. LangChainのRAGチェーンで利用する例 +### 3.3. LangChain RAGチェーンでの利用 + +上記いずれかの方法で初期化された hybrid_retriever は、標準的なLangChainのRAGチェーンに組み込むことができます。 

```Python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

+# ChatOpenAIの代わりに、利用したいLLMモデルを指定してください。
+# 例: from langchain_community.chat_models import ChatAnthropic
+# 例: from langchain_huggingface import HuggingFacePipeline
+
+# 前のセクションで初期化された hybrid_retriever_via_system または hybrid_retriever_direct を使用
+# ここでは hybrid_retriever_direct を使う例とします
+
qa_chain = RetrievalQA.from_chain_type(
-    llm=ChatOpenAI(),
-    retriever=hybrid_retriever,
+    llm=ChatOpenAI(model="gpt-3.5-turbo"), # モデル名は適宜変更
+    retriever=hybrid_retriever_direct, # ここに作成したHybridRetrieverインスタンスを指定
    return_source_documents=True
)

-result = qa_chain({"query": "検索したい内容"})
-print(result["result"])
+response = qa_chain.invoke({"query": "LangChainにおけるハイブリッド検索とは何ですか？"})
+print("\n--- RAGチェーンからの回答 ---")
+print(response["result"])
+print("\n--- 参照されたソースドキュメント ---")
+for doc in response["source_documents"]:
+    print(f"ID: {doc.metadata.get('id', 'N/A')}, Content: {doc.page_content[:50]}...")
```

## 5. 注意点 -- Chroma/BM25のデータベース構築はmain.pyやStreamlitアプリで事前に行ってください。 -- Retrieverのget_relevant_documentsは重複排除済みのリストを返します。 -- BM25リトリーバーが未構築の場合はベクトル検索のみで動作します。 +## 4. 注意点とヒント +- データベース構築: Chromaデータベースの構築やBM25リトリーバーの準備（特に日本語テキストの形態素解析など）は、HybridRetriever を利用する前に行ってください。main.py の HybridSearchSystem はその一例です。 +- 重複排除: get_relevant_documentsメソッドは、ベクトル検索とBM25検索の結果をマージし、重複するドキュメントをID（doc.metadata['id']を想定）に基づいて排除したリストを返します。適切なメタデータIDをドキュメントに付与してください。 +- BM25リトリーバーの有無: HybridRetriever 初期化時に bm25_retriever に None を指定するか、またはBM25リトリーバーがドキュメントを返さなかった場合、ベクトル検索のみの結果が利用されます。逆も同様ですが、通常は両方のリトリーバーを提供することが推奨されます。 +- SudachiPyの利用: BM25検索で日本語の精度を向上させるためには、SudachiPy と適切な辞書（例: sudachidict_core）を用いた形態素解析をドキュメント登録前およびクエリ実行前に行うことを強く推奨します。main.py の JapaneseBM25Retriever がその実装例です。

-## 6. カスタマイズ -- HybridRetrieverのweights引数でベクトル/BM25の重みを調整できます。 -- マージロジックやチャンク分割は用途に応じて拡張可能です。 +## 5. カスタマイズ +- weights引数: HybridRetriever の初期化時に weights=[ベクトル検索の重み, BM25検索の重み] (例: [0.5, 0.5] や [0.7, 0.3]) を指定することで、各検索手法のスコアへの寄与度を調整できます。デフォルトは [0.5, 0.5] です。 +- 検索結果の取得数 (k):

- ベクトルリトリーバー: vector_store.as_retriever(search_kwargs={"k": N}) のように、as_retriever メソッドの search_kwargs で取得数を指定します。
- BM25リトリーバー: bm25_retriever.k = N のように、インスタンスの k 属性に直接設定します。
- HybridRetriever は、これら各リトリーバーから指定された数のドキュメントを取得し、マージ処理を行います。 +- チャンク分割: ドキュメントのチャンク分割戦略は、検索品質に大きく影響します。データソースや用途に応じて最適な分割方法を検討し、ChromaやBM25へのデータ登録前に適用してください。 +- マージロジック: HybridRetriever は、各リトリーバーからの結果をスコアに基づいてマージします（現在はReciprocal Rank Fusion (RRF) に似た手法を採用）。より高度な、または特定用途に合わせたマージ戦略が必要な場合は、tools/hybrid_retriever.py の HybridRetriever クラスを拡張するか、参考に独自のものを実装することを検討してください。

## 6. main.py (HybridSearchSystem) について
- main.py に含まれる HybridSearchSystem クラスは、このハイブリッド検索モジュールの利用方法を示すためのサンプル実装です。データのロード、Chromaデータベースのセットアップ、JapaneseBM25Retriever（SudachiPyを利用）の初期化など、一連の処理をまとめています。
- ご自身のアプリケーションに HybridRetriever を組み込む際には、HybridSearchSystem の実装を参考にしつつ、データソースや前処理、リトリーバーの初期化方法などを適宜カスタマイズしてください。HybridRetriever 自体は、HybridSearchSystem から独立して利用可能です（「3.2. 個別のリトリーバーコンポーネントからの直接初期化」参照）。



