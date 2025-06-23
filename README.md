# Enhanced Raman Spectrum Analysis Tool with Hugging Face Mistral

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

高度なラマンスペクトル解析ツールで、Hugging Face Mistralモデルを使用したAI解析とRAG（Retrieval-Augmented Generation）機能を提供します。

## 🚀 主な機能

### 📊 スペクトル解析
- **多様なファイル形式対応**: RamanEye、Wasatch ENLIGHTEN、Eagleデータ
- **高度なピーク検出**: 2次微分 + prominence判定による精密な検出
- **インタラクティブ操作**: クリックによる手動ピーク追加・除外
- **グリッドサーチ最適化**: パラメータ自動調整

### 🤖 AI解析
- **Hugging Face Mistral**: 最新の大規模言語モデルによる解析
- **RAG機能**: 論文データベースからの関連情報検索
- **ストリーミング応答**: リアルタイムでの解析結果表示
- **日本語対応**: 完全日本語での解析結果

### 📈 可視化
- **Plotlyによるインタラクティブグラフ**: 3段階プロット（スペクトル、微分、prominence）
- **リアルタイム更新**: パラメータ変更時の即座な反映
- **複数ファイル対応**: 同時に複数スペクトルの解析

## 🛠️ インストール

### 前提条件
- Python 3.8以上
- CUDA対応GPU（推奨、CPU動作も可能）
- 8GB以上のRAM/VRAM

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-username/raman-analysis-tool.git
cd raman-analysis-tool
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. Hugging Face Tokensの設定（オプション）
プライベートモデルを使用する場合：
```bash
huggingface-cli login
```

## 🖥️ 使用方法

### アプリケーション起動
```bash
streamlit run app.py
```

### 基本的な操作手順

1. **Mistralモデル選択**
   - サイドバーから使用するMistralモデルを選択
   - 推奨: `mistralai/Mistral-7B-Instruct-v0.2`

2. **論文データベース構築**
   - 参考文献PDFをアップロード（複数可）
   - 「論文データベース構築」ボタンをクリック

3. **スペクトルファイルアップロード**
   - CSV/TXTファイルを選択
   - 対応形式: RamanEye、Wasatch、Eagle

4. **ピーク検出**
   - パラメータを調整
   - 「ピーク検出を実行」ボタンをクリック

5. **手動調整**
   - グラフ上をクリックしてピーク追加/除外
   - グリッドサーチで最適化

6. **AI解析**
   - 「AI解析を実行」ボタンをクリック
   - Mistralによる詳細な考察を取得

## 📁 ファイル構造

```
raman-analysis-tool/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存関係
├── README.md             # このファイル
├── tmp_uploads/          # アップロードファイル一時保存
└── sample_data/          # サンプルデータ（オプション）
```

## 🔧 設定可能なパラメータ

### ピーク検出パラメータ
- **波数範囲**: 解析する波数範囲の指定
- **ベースラインパラメータ**: airPLS法のパラメータ
- **2次微分平滑化**: Savitzky-Golay filterのウィンドウサイズ
- **prominence閾値**: ピーク卓立度の閾値

### AI解析パラメータ
- **モデル選択**: 使用するMistralモデル
- **最大トークン数**: 生成する応答の最大長
- **温度**: 創造性の制御パラメータ

## 📊 サポートファイル形式

### スペクトルデータ
- **RamanEye**: 新旧フォーマット対応
- **Wasatch ENLIGHTEN**: 標準出力形式
- **Eagle**: 転置形式データ
- **汎用CSV/TXT**: カスタムフォーマット

### 参考文献
- **PDF**: 論文、レポート
- **DOCX**: Word文書
- **TXT**: プレーンテキスト

## 🤖 AI機能の詳細

### Mistralモデル
- **Mistral-7B-Instruct-v0.2**: バランスの良い性能
- **Mistral-7B-Instruct-v0.1**: 軽量版
- **Mixtral-8x7B-Instruct-v0.1**: 高性能版（要大容量メモリ）

### RAG機能
- **ベクトル検索**: FAISS基盤の高速検索
- **文書分割**: 最適なチャンクサイズでの分割
- **類似度計算**: コサイン類似度による関連文書抽出

## 🎯 使用例

### 1. ポリマー分析
```python
# グラファイト系材料の特徴的ピーク
G-band: ~1580 cm⁻¹
D-band: ~1350 cm⁻¹
```

### 2. 無機材料分析
```python
# 酸化物の特徴的ピーク
Si-O: 400-1200 cm⁻¹
Ti-O: 400-800 cm⁻¹
```

## 🔧 トラブルシューティング

### よくある問題

1. **GPUメモリ不足**
   ```bash
   # 解決策: より小さなモデルを使用
   # または環境変数でCPU強制使用
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **依存関係エラー**
   ```bash
   # 解決策: 仮想環境の使用
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # または
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

3. **ファイル読み込みエラー**
   - 文字エンコーディングの確認（UTF-8, Shift-JIS対応）
   - ファイル形式の確認

## 📈 パフォーマンス最適化

### GPU使用時
- **メモリ使用量**: 4-16GB VRAM
- **処理速度**: 10-30秒/解析

### CPU使用時
- **メモリ使用量**: 8-32GB RAM
- **処理速度**: 1-5分/解析

## 🤝 貢献

1. Forkしてください
2. Feature branchを作成してください (`git checkout -b feature/AmazingFeature`)
3. 変更をcommitしてください (`git commit -m 'Add some AmazingFeature'`)
4. Branchにpushしてください (`git push origin feature/AmazingFeature`)
5. Pull Requestを作成してください

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルを参照してください。

## 👥 開発者

- **開発者**: hiroy
- **連絡先**: [your-email@example.com]
- **GitHub**: [https://github.com/your-username]

## 🙏 謝辞

- Hugging Face Transformersチーム
- Streamlitコミュニティ
- オープンソースコミュニティ

## 📊 統計情報

![GitHub stars](https://img.shields.io/github/stars/your-username/raman-analysis-tool?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/raman-analysis-tool?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/raman-analysis-tool)

---

🚀 **高度なラマンスペクトル解析を、AIの力で。**
