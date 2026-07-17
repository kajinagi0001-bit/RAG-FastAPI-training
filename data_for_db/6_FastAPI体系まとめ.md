# FastAPI体系まとめ

## 1. FastAPIとは何か

FastAPIは、PythonでWeb APIを作るためのフレームワークである。

特徴：

- Pythonの型ヒントと相性がよい
- API仕様書を自動生成できる
- 非同期処理に対応している
- JSON APIを作りやすい
- 機械学習モデルのAPI化に向いている
- Flaskより構造化しやすく、Djangoより軽量

画像認識モデル、LLMアプリ、RAG、WebバックエンドなどのAPI化に適している。

---

## 2. FastAPIを学ぶ意味

FastAPIを使えると、以下ができる。

- PythonコードをAPIとして公開する
- 画像認識モデルを推論APIにする
- RAGアプリのバックエンドを作る
- WebフロントエンドとAIモデルを接続する
- DBと連携したアプリを作る
- DockerでAPIサーバーを動かす
- クラウドにAI APIをデプロイする

AI開発では、モデルを作るだけでなく、他の人やシステムが使える形にする力が重要である。

---

## 3. Web APIの基本

APIとは、他のプログラムから機能やデータを利用するための入口である。

```text
クライアント
  ↓ HTTPリクエスト
APIサーバー
  ↓ HTTPレスポンス
クライアント
```

画像認識APIの例：

```text
POST /predict
  ↓
画像を送る
  ↓
推論結果をJSONで返す
```

---

## 4. インストールと最小アプリ

```bash
pip install fastapi uvicorn
```

`main.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}
```

起動：

```bash
uvicorn main:app --reload
```

アクセス：

```text
http://localhost:8000
http://localhost:8000/docs
```

---

## 5. Uvicornとは何か

Uvicornは、FastAPIアプリを実行するためのASGIサーバーである。
**ASGI**とは、Pythonの非同期Webアプリを動かすための仕様である。

```text
FastAPIアプリ
  ↓
Uvicorn
  ↓
HTTP通信
```

開発：

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Dockerや外部公開：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 6. ルーティング

URLと処理関数を対応付けることをルーティングという。

```python
@app.get("/users")
def get_users():
    return [{"id": 1, "name": "Taro"}]

@app.post("/users")
def create_user():
    return {"message": "created"}
```

| メソッド | FastAPI | 用途 |
|---|---|---|
| GET | `@app.get()` | 取得 |
| POST | `@app.post()` | 作成・送信 |
| PUT | `@app.put()` | 全体更新 |
| PATCH | `@app.patch()` | 部分更新 |
| DELETE | `@app.delete()` | 削除 |

---

## 7. パスパラメータとクエリパラメータ

### パスパラメータ

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
```

```text
GET /users/10
```

### クエリパラメータ

```python
@app.get("/search")
def search(q: str, limit: int = 10):
    return {"q": q, "limit": limit}
```

```text
GET /search?q=cat&limit=5
```

---

## 8. リクエストボディとPydantic

POSTなどでJSONを受け取るには、Pydanticモデルを使う。

```python
from pydantic import BaseModel

class UserCreate(BaseModel):
    name: str
    age: int

@app.post("/users")
def create_user(user: UserCreate):
    return {"name": user.name, "age": user.age}
```

Pydanticは、データの型チェックやバリデーションを行うライブラリである。

---

## 9. レスポンスとステータスコード

Pythonの辞書やリストを返すと、自動的にJSONに変換される。

```python
@app.get("/items")
def get_items():
    return [{"id": 1, "name": "apple"}]
```

ステータスコード指定：

```python
from fastapi import status

@app.post("/users", status_code=status.HTTP_201_CREATED)
def create_user():
    return {"message": "created"}
```

代表的なステータスコード：

| コード | 意味 |
|---|---|
| 200 | 成功 |
| 201 | 作成成功 |
| 400 | 不正なリクエスト |
| 401 | 認証が必要 |
| 403 | 権限がない |
| 404 | 見つからない |
| 500 | サーバー内部エラー |

---

## 10. エラーハンドリング

```python
from fastapi import HTTPException

@app.get("/users/{user_id}")
def get_user(user_id: int):
    if user_id != 1:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": 1, "name": "Taro"}
```

APIでは、エラー内容を適切なステータスコードで返すことが重要である。

---

## 11. ファイルアップロード

```python
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content)
    }
```

`UploadFile` は、大きなファイルを扱いやすい形式で受け取るための仕組みである。

---

## 12. 画像推論APIの基本形

```python
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 前処理・モデル推論を行う
    result = {
        "label": "cat",
        "score": 0.98
    }

    return result
```

実際には以下を追加する。

- 画像形式チェック
- ファイルサイズ制限
- 前処理
- モデル読み込み
- 推論
- 後処理
- エラーハンドリング

---

## 13. モデル読み込み

悪い例：

```python
@app.post("/predict")
def predict():
    model = load_model()
```

リクエストごとにモデルを読み込むため遅い。

良い例：

```python
app = FastAPI()
model = load_model()

@app.post("/predict")
def predict():
    result = model(...)
    return result
```

起動時に1回だけモデルを読み込む。

---

## 14. lifespan

起動時・終了時の処理にはlifespanを使う。

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None

app = FastAPI(lifespan=lifespan)
```

用途：

- モデル読み込み
- DB接続
- ベクトルDB接続
- キャッシュ初期化
- 終了時のリソース解放

---

## 15. 非同期処理

```python
@app.get("/async")
async def async_endpoint():
    return {"message": "async"}
```

非同期処理は、I/O待ちが多い処理に向いている。

例：

- 外部API呼び出し
- DBアクセス
- ファイル読み込み
- ネットワーク通信

CPUやGPUを長時間使う推論は、単純に `async` にしても速くなるとは限らない。

---

## 16. CORS

フロントエンドとAPIサーバーのオリジンが異なる場合、CORS設定が必要になる。

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

本番では必要なオリジンだけ許可する。

---

## 17. 依存性注入

依存性注入とは、共通処理をエンドポイントに渡す仕組みである。

```python
from fastapi import Depends

def get_token():
    return "token"

@app.get("/me")
def read_me(token: str = Depends(get_token)):
    return {"token": token}
```

用途：

- 認証
- DBセッション
- 設定値
- 権限チェック
- 共通バリデーション

---

## 18. 認証

簡単なAPIキー認証の例：

```python
import os
from fastapi import Header, HTTPException, Depends

API_KEY = os.environ["API_KEY"]

def verify_api_key(x_api_key: str = Header()):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/secure")
def secure_endpoint(_: None = Depends(verify_api_key)):
    return {"message": "ok"}
```

本番では、APIキーをコードに直接書かず、環境変数から読む。

---

## 19. DB連携

構成：

```text
クライアント
  ↓
FastAPI
  ↓
DB
```

よく使うDB：

- SQLite
- PostgreSQL
- MySQL
- Redis
- ChromaDB
- Neo4j

Pythonでは、SQLAlchemyやSQLModelを使うことが多い。

---

## 20. ディレクトリ構成例

```text
project/
├── app/
│   ├── main.py
│   ├── api/
│   ├── schemas/
│   ├── models/
│   ├── services/
│   ├── core/
│   └── db/
├── tests/
├── requirements.txt
└── Dockerfile
```

| ディレクトリ | 内容 |
|---|---|
| api | ルーティング |
| schemas | Pydanticモデル |
| models | DBモデル |
| services | ビジネスロジック |
| core | 設定 |
| db | DB接続 |
| tests | テスト |

---

## 21. Docker化

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t fastapi-app .
docker run -p 8000:8000 fastapi-app
```

---

## 22. テスト

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
```

```bash
pytest
```

API開発では、正常系だけでなく異常系もテストする。

---

## 23. 本番運用

本番では、開発用の `--reload` は使わない。

```text
ユーザー
  ↓ HTTPS
Nginx
  ↓ HTTP
Uvicorn / Gunicorn
  ↓
FastAPI
  ↓
DB
```

考えること：

- HTTPS
- Nginx
- ワーカー数
- タイムアウト
- ログ
- 監視
- 環境変数
- セキュリティ
- DBバックアップ
- CORS設定
- ファイルサイズ制限

---

## 24. AI推論APIで注意すること

| 項目 | 注意点 |
|---|---|
| モデル読み込み | 起動時に1回だけ行う |
| GPUメモリ | 同時リクエストで不足する可能性 |
| 画像サイズ | 大きすぎる画像を制限する |
| ファイル形式 | JPEG/PNGなどを検証する |
| 前処理 | 学習時と同じ変換を行う |
| タイムアウト | 推論が長すぎる場合に備える |
| バッチ化 | 複数入力をまとめると高速化できる場合がある |
| ログ | 処理時間やエラーを記録する |
| セキュリティ | 不正ファイル対策を行う |

---

## 25. よくあるエラー

| エラー | 意味 | 対応 |
|---|---|---|
| 422 | リクエスト形式がPydanticモデルと違う | JSONキー・型を確認 |
| CORS Error | ブラウザが異なるオリジン通信を制限 | CORS設定 |
| Connection refused | サーバー未起動・ポート違い | `curl`, `lsof` |
| 500 | サーバー内部例外 | ログ確認 |
| ModuleNotFoundError | Pythonパッケージ不足 | requirements確認 |

---

## 26. 学習ロードマップ

```text
1. HTTPとAPIの基礎
2. FastAPI最小アプリ
3. GET / POST
4. パス・クエリ・ボディ
5. Pydantic
6. エラーハンドリング
7. ファイルアップロード
8. 画像推論API
9. CORS
10. DB連携
11. Docker化
12. テスト
13. Nginxと本番運用
14. 認証
15. 監視・ログ
```

---

## 27. まとめ

FastAPIは、PythonでWeb APIを作るための強力なフレームワークである。

| 概念 | 説明 |
|---|---|
| ルーティング | URLと処理関数の対応 |
| パスパラメータ | URL内の変数 |
| クエリパラメータ | URLの `?` 以降の値 |
| リクエストボディ | POSTなどで送るJSON |
| Pydantic | 型チェック・バリデーション |
| UploadFile | ファイルアップロード |
| CORS | 異なるオリジン通信の制御 |
| Depends | 依存性注入 |
| Uvicorn | FastAPIを動かすASGIサーバー |
| Docker | 実行環境のコンテナ化 |

FastAPIを学ぶと、PythonコードやAIモデルを、他の人やシステムから利用できるAPIとして提供できるようになる。
