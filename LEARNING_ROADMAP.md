# Learning Roadmap

このプロジェクトを一から読むときは、APIの入口から入り、内部処理へ少しずつ潜っていく順番がおすすめです。

## 1. 全体構成

読むファイル:

```text
README.md
PLAN.md
app/main.py
```

理解すること:

```text
どんなAPIがあるか
どの機能が実装済みか
FastAPIの入口がどこか
```

まずは `app/main.py` の `@app.get` / `@app.post` を眺めて、URLと関数の対応を確認します。

## 2. DBの土台

読むファイル:

```text
app/database.py
app/models.py
app/db_schema.py
```

理解すること:

```text
SQLiteに接続する流れ
SQLAlchemyのModel
Document / Chunk / RagRun / Memory の関係
既存DBに列を追加する仕組み
```

特に重要なのは `app/models.py` です。これは、このプロジェクトで保存するデータの設計図です。

## 3. ドキュメント登録

読むファイル:

```text
app/ingestion.py
app/chunking.py
app/tools.py
```

追う流れ:

```text
POST /documents/upload
  -> extract_upload_text
  -> create_document_with_chunks
  -> chunk_text
  -> embedding
  -> DB保存
```

ここで、Markdown、テキスト、PDFがどのようにDBへ入るかを理解します。

## 4. Embeddingと検索

読むファイル:

```text
app/embedding.py
app/retrieval.py
app/retrieval_service.py
```

追う流れ:

```text
質問
  -> embedding化
  -> DB内chunk embeddingとcosine similarity
  -> 上位chunkを返す
```

ここがRAGのRetrieval部分です。まずはローカルembeddingの流れを理解し、その後OpenAI embeddingの分岐を見ると分かりやすいです。

## 5. 通常RAG

読むファイル:

```text
app/tools.py
app/llm.py
```

追う流れ:

```text
run_rag_chat
  -> search_knowledge_base
  -> answer_with_context
  -> generate_answer
  -> log_rag_run
```

まず `normal RAG` を理解してください。ここが分かると、会話履歴、agent、tool-calling、評価機能がすべて繋がります。

## 6. 会話履歴

読むファイル:

```text
app/main.py
app/schemas.py
```

追う流れ:

```text
POST /conversations
POST /conversations/{id}/chat
  -> Message保存
  -> recent history取得
  -> run_rag_chat(history付き)
```

ここでは、一回きりの質問と会話形式の質問の違いを理解します。

## 7. Agent

読むファイル:

```text
app/agent.py
```

追う流れ:

```text
plan
  -> search_memories
  -> search_knowledge_base
  -> decide_retry_search
  -> answer_with_context
  -> log_rag_run
```

このAgentは、Python側で判断ロジックを書いているAgentです。検索結果が弱ければ `top_k` を増やして再検索し、memoryも回答生成に渡します。

## 8. Tool-calling Agent

読むファイル:

```text
app/tool_calling_agent.py
```

追う流れ:

```text
TOOL_DEFINITIONS
  -> OpenAI Responses API
  -> execute_tool_call
  -> tool結果をLLMへ戻す
  -> final answer
  -> tool call trace保存
```

ここが一番エージェントらしい部分です。ただし難しいので、先に `app/agent.py` を理解してから読むのがおすすめです。

## 9. 評価機能

読むファイル:

```text
app/judge.py
scripts/eval_retrieval.py
scripts/eval_answers.py
scripts/eval_tool_calls.py
scripts/eval_memories.py
scripts/eval_agent_runs.py
```

見る観点:

```text
検索品質
回答品質
tool call品質
memory品質
run_type比較
```

ここで、動くだけのRAGから改善できるRAGへ進みます。

## 10. UI

読むファイル:

```text
app/chat_ui.py
app/dashboard.py
```

追う流れ:

```text
GET /
  -> HTML表示
  -> fetchでAPI呼び出し
  -> answer / sources / stepsを描画

GET /dashboard
  -> DB集計
  -> HTML表示
```

UIは便利ですが、内部理解としては最後で十分です。まずバックエンドの流れを理解してから読むと、画面がどのAPIを使っているかが自然に分かります。

## 推奨順の短縮版

```text
app/main.py
  -> app/database.py / app/models.py
  -> app/ingestion.py / app/chunking.py
  -> app/embedding.py / app/retrieval_service.py
  -> app/tools.py / app/llm.py
  -> app/agent.py
  -> app/tool_calling_agent.py
  -> app/judge.py / scripts
  -> app/chat_ui.py / app/dashboard.py
```

最初の目標は、`/chat` がどう動くかを自分で説明できるようになることです。

