# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 09:02:35 2025

@author: hiroy

Enhanced Raman Spectrum Analysis Tool with HuggingFace Mistral and RAG functionality
"""

import time
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import scipy.signal as signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags

from streamlit_plotly_events import plotly_events

# Additional imports for LLM and RAG functionality
import os
import glob
import PyPDF2
import docx
from datetime import datetime
from typing import List, Dict, Optional
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
import numpy as np

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

# RAG機能のクラス定義（transformersベース）
class SimpleRAGSystem:
    def __init__(self, embedding_model_name='cl-tohoku/bert-base-japanese'):
        """
        RAGシステムの初期化（transformersライブラリ使用）

        Args:
            embedding_model_name: 使用する埋め込みモデル名
        """
        self.embedding_model_name = embedding_model_name
        self.tokenizer = None
        self.model = None
        self.vector_db = None
        self.documents = []
        self.document_metadata = []
        self.embedding_dim = None
        self._model_loaded = False

    def extract_text_from_file(self, file_path: str) -> str:
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_ext == '.docx':
                return self._extract_from_docx(file_path)
            elif file_ext == '.txt':
                return self._extract_from_txt(file_path)
            else:
                return ""
        except Exception as e:
            st.error(f"ファイル {file_path} の読み込みエラー: {e}")
            return ""

    def _extract_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"PDF読み込みエラー: {e}")
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"DOCX読み込みエラー: {e}")
            return ""

    def _extract_from_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='shift_jis') as file:
                    return file.read()
            except Exception as e:
                st.error(f"TXT読み込みエラー: {e}")
                return ""
        except Exception as e:
            st.error(f"TXT読み込みエラー: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _load_embedding_model(self):
        if not self._model_loaded:
            try:
                with st.spinner("埋め込みモデルを読み込み中..."):
                    cache_dir = os.path.join(os.getcwd(), "model_cache")
                    os.makedirs(cache_dir, exist_ok=True)

                    import fugashi  # Mecab用依存を明示的にロード

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.embedding_model_name,
                        cache_dir=cache_dir,
                        use_fast=False
                    )
                    self.model = AutoModel.from_pretrained(
                        self.embedding_model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float32,
                        device_map="cpu"
                    )
                    self.model.eval()
                    self._model_loaded = True
                    st.success("✅ 埋め込みモデルの読み込み完了")
            except Exception as e:
                st.error(f"埋め込みモデルの読み込みに失敗しました: {e}")
                st.info("💡 RAG機能を無効にしてください")
                return False
        return True

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not self._model_loaded:
            return np.array([])
        embeddings = []
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings) if embeddings else np.array([])

    def build_vector_database(self, folder_path: str):
        if not self._load_embedding_model():
            return
        if not os.path.exists(folder_path):
            st.error(f"指定されたフォルダが存在しません: {folder_path}")
            return
        file_patterns = ['*.pdf', '*.docx', '*.txt']
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(os.path.join(folder_path, pattern)))
        if not files:
            st.warning("指定されたフォルダに論文ファイルが見つかりません。")
            return
        st.info(f"論文ファイル {len(files)} 件を処理中...")
        all_chunks = []
        all_metadata = []
        progress_bar = st.progress(0)
        for i, file_path in enumerate(files):
            text = self.extract_text_from_file(file_path)
            if text:
                chunks = self.chunk_text(text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'filename': os.path.basename(file_path),
                        'filepath': file_path,
                        'chunk_text': chunk[:100] + "..." if len(chunk) > 100 else chunk
                    })
            progress_bar.progress((i + 1) / len(files))
        if not all_chunks:
            st.error("処理可能なテキストが見つかりませんでした。")
            return
        st.info("埋め込みベクトルを生成中...")
        progress_bar2 = st.progress(0)
        embeddings = self._encode_texts(all_chunks)
        progress_bar2.progress(1.0)
        if len(embeddings) == 0:
            st.error("埋め込みベクトルの生成に失敗しました。")
            return
        self.embedding_dim = embeddings.shape[1]
        self.vector_db = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings.astype(np.float32))
        self.vector_db.add(embeddings.astype(np.float32))
        self.documents = all_chunks
        self.document_metadata = all_metadata
        st.success(f"ベクトルデータベースの構築完了: {len(all_chunks)} チャンク")

    def search_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.vector_db is None:
            return []
        if not self._load_embedding_model():
            return []
        try:
            query_embeddings = self._encode_texts([query])
            if len(query_embeddings) == 0:
                return []
            faiss.normalize_L2(query_embeddings.astype(np.float32))
            scores, indices = self.vector_db.search(query_embeddings.astype(np.float32), top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.document_metadata[idx],
                        'similarity_score': float(score)
                    })
            return results
        except Exception as e:
            st.error(f"文書検索中にエラーが発生しました: {e}")
            return []


class SimpleLLM:
    """
    シンプルなLLMクラス（軽量版）
    """
    def __init__(self, model_name="cyberagent/open-calm-small"):
        self.model_name = model_name
        self.pipeline = None
        self._model_loaded = False

    def _load_model(self):
        if self._model_loaded:
            return True

        try:
            with st.spinner(f"言語モデル ({self.model_name}) をロード中..."):
                cache_dir = os.path.join(os.getcwd(), "model_cache")
                os.makedirs(cache_dir, exist_ok=True)

                tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
                model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=cache_dir)

                
                generator = pipeline(
                    "text-generation",
                    model="cyberagent/open-calm-small",
                    tokenizer="cyberagent/open-calm-small",
                    device=-1  # CPU
                )

                self._model_loaded = True
                st.success(f"✅ 言語モデルのロード完了")
                return True

        except Exception as e:
            st.error(f"言語モデルのロードに失敗しました: {e}")
            st.info("💡 解決策: \n1. モデル名のスペル確認\n2. ネットワーク確認\n3. requirementsからsentencepieceを外す")
            return False
    def generate_response(self, prompt: str, max_tokens: int = 256) -> str:
        """
        プロンプトに対する応答を生成
        
        Args:
            prompt: 入力プロンプト
            max_tokens: 最大トークン数
            
        Returns:
            生成された応答テキスト
        """
        if not self._load_model():
            return "⚠️ モデルが利用できません。基本的なピーク解析結果のみ表示しています。"
        
        try:
            # プロンプトを日本語解析用に最適化
            formatted_prompt = f"""以下のラマンスペクトル解析データを基に、試料の成分を推定してください。

{prompt}

回答は日本語で、以下の観点から詳しく説明してください：
1. 各ピークの化学的帰属
2. 推定される試料の種類
3. 根拠となるピーク位置の解釈

回答:"""
            
            # テキスト生成
            if "DialoGPT" in self.model_name:
                response = self.pipeline(
                    formatted_prompt,
                    max_length=len(formatted_prompt) + max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                generated_text = response[0]['generated_text']
                # プロンプト部分を除去
                result = generated_text[len(formatted_prompt):].strip()
            else:
                response = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                result = response[0]['generated_text'].strip()
            
            return result if result else "解析を実行しましたが、具体的な推定結果を生成できませんでした。"
            
        except Exception as e:
            return f"⚠️ 応答生成中にエラーが発生しました: {e}"
    
    def generate_stream_response(self, prompt: str, max_tokens: int = 256):
        full_response = self.generate_response(prompt, max_tokens)
    
        if not full_response:
            yield "⚠️ 応答が生成されませんでした。プロンプトまたはモデルの設定を確認してください。"
            return
    
        for i in range(0, len(full_response), 5):
            chunk = full_response[i:i+5]
            yield chunk
            time.sleep(0.03)

class RamanSpectrumAnalyzer:
    def generate_analysis_prompt(
        self,
        peak_data: List[Dict],
        relevant_docs: List[Dict],
        user_hint: Optional[str] = None
    ) -> str:
        """
        ラマンスペクトル解析のためのプロンプトを生成

        Args:
            peak_data: ピーク情報のリスト
            relevant_docs: 関連文献情報（RAG結果）
            user_hint: ユーザー補足コメント（任意）

        Returns:
            LLM用解析プロンプト文字列
        """

        def format_peaks(peaks: List[Dict]) -> str:
            header = "【検出ピーク一覧】"
            lines = [
                f"{i+1}. 波数: {p.get('wavenumber', 0):.1f} cm⁻¹, "
                f"強度: {p.get('intensity', 0):.3f}, "
                f"prominence: {p.get('prominence', 0):.3f}, "
                f"種類: {'自動検出' if p.get('type') == 'auto' else '手動追加'}"
                for i, p in enumerate(peaks)
            ]
            return "\n".join([header] + lines)

        def format_reference_excerpts(docs: List[Dict]) -> str:
            header = "【引用文献の抜粋と要約】"
            lines = []
            for i, doc in enumerate(docs, 1):
                title = doc.get("metadata", {}).get("filename", f"文献{i}")
                page = doc.get("metadata", {}).get("page")
                summary = doc.get("page_content", "").strip()
                lines.append(f"\n--- 引用{i} ---")
                lines.append(f"出典ファイル: {title}")
                if page is not None:
                    lines.append(f"ページ番号: {page}")
                lines.append(f"抜粋内容:\n{summary}")
            return "\n".join([header] + lines)

        def format_doc_summaries(docs: List[Dict], preview_length: int = 300) -> str:
            header = "【文献の概要（類似度付き）】"
            lines = []
            for i, doc in enumerate(docs, 1):
                filename = doc.get("metadata", {}).get("filename", f"文献{i}")
                similarity = doc.get("similarity_score", 0.0)
                text = doc.get("text") or doc.get("page_content") or ""
                lines.append(
                    f"文献{i} (類似度: {similarity:.3f})\n"
                    f"ファイル名: {filename}\n"
                    f"冒頭抜粋: {text.strip()[:preview_length]}...\n"
                )
            return "\n".join([header] + lines)

        # プロンプト本文の構築
        sections = [
            "以下は、ラマンスペクトルで検出されたピーク情報です。",
            "これらのピークに基づき、試料の成分や特徴について推定してください。",
            "なお、文献との比較においてはピーク位置が±5cm⁻¹程度ずれることがあります。",
            "そのため、±5cm⁻¹以内の差であれば一致とみなして解析を行ってください。\n"
        ]

        if user_hint:
            sections.append(f"【ユーザーによる補足情報】\n{user_hint}\n")

        if peak_data:
            sections.append(format_peaks(peak_data))
        if relevant_docs:
            sections.append(format_reference_excerpts(relevant_docs))
            sections.append(format_doc_summaries(relevant_docs))

        sections.append(
            "これらを参考に、試料に含まれる可能性のある化合物や物質構造、特徴について詳しく説明してください。\n"
            "出力は日本語でお願いします。\n"
            "## 解析の観点:\n"
            "1. 各ピークの化学的帰属とその根拠\n"
            "2. 試料の可能な組成や構造\n"
            "3. 文献情報との比較・対照\n\n"
            "詳細で科学的根拠に基づいた考察を日本語で提供してください。"
        )

        return "\n".join(sections)
    
    def _generate_basic_analysis(self, peak_data: List[Dict]) -> str:
        """
        基本的なピーク解析を生成（AI不使用）
        
        Args:
            peak_data: ピーク情報のリスト
            
        Returns:
            基本解析テキスト
        """
        # 一般的なラマンピークの解釈テーブル
        common_peaks = {
            (400, 500): "金属酸化物の結合振動",
            (500, 600): "S-S結合、金属-酸素結合",
            (600, 800): "C-S結合、芳香環の変角振動",
            (800, 1000): "C-C結合の伸縮振動",
            (1000, 1200): "C-O結合、C-N結合の伸縮振動",
            (1200, 1400): "C-H結合の変角振動",
            (1400, 1600): "C=C結合の伸縮振動（芳香環）",
            (1600, 1800): "C=O結合の伸縮振動",
            (2800, 3000): "C-H結合の伸縮振動（アルキル）",
            (3000, 3200): "C-H結合の伸縮振動（芳香環）",
            (3200, 3600): "O-H結合の伸縮振動",
        }
        
        analysis_parts = ["## 検出ピークの化学的解釈\n"]
        
        for i, peak in enumerate(peak_data, 1):
            wavenumber = peak['wavenumber']
            intensity = peak['intensity']
            peak_type = '自動検出' if peak['type'] == 'auto' else '手動追加'
            
            # ピーク解釈を検索
            interpretations = []
            for (min_wn, max_wn), description in common_peaks.items():
                if min_wn <= wavenumber <= max_wn:
                    interpretations.append(description)
            
            analysis_parts.append(f"**ピーク {i} ({wavenumber:.1f} cm⁻¹, {peak_type})**")
            if interpretations:
                analysis_parts.append(f"- 推定結合: {', '.join(interpretations)}")
            else:
                analysis_parts.append("- 推定結合: 特定の結合の同定が困難")
            
            # 強度による解釈
            if intensity > 0.8:
                analysis_parts.append("- 強度: 非常に強い（主要成分の特徴的ピーク）")
            elif intensity > 0.5:
                analysis_parts.append("- 強度: 強い（重要な構造成分）")
            elif intensity > 0.2:
                analysis_parts.append("- 強度: 中程度（副次的成分）")
            else:
                analysis_parts.append("- 強度: 弱い（微量成分または雑音）")
            
            analysis_parts.append("")
        
        # 全体的な傾向分析
        analysis_parts.append("## 試料の特徴推定\n")
        
        wavenumbers = [p['wavenumber'] for p in peak_data]
        
        # 有機物vs無機物の判定
        organic_count = sum(1 for wn in wavenumbers if 800 <= wn <= 3600)
        inorganic_count = sum(1 for wn in wavenumbers if 200 <= wn <= 800)
        
        if organic_count > inorganic_count:
            analysis_parts.append("- **有機化合物の特徴が強い**: C-H, C=C, C=O等の有機結合が多数検出")
        elif inorganic_count > organic_count:
            analysis_parts.append("- **無機化合物の特徴が強い**: 金属酸化物や無機結合が主要")
        else:
            analysis_parts.append("- **有機・無機混合物の可能性**: 両方の特徴を含有")
        
        # 特定の化合物群の推定
        if any(1300 <= wn <= 1400 for wn in wavenumbers):
            analysis_parts.append("- **炭素系材料の可能性**: グラファイト系材料のD-bandが検出される可能性")
        
        if any(1580 <= wn <= 1590 for wn in wavenumbers):
            analysis_parts.append("- **グラファイト系材料**: G-bandが検出される可能性")
        
        if any(2800 <= wn <= 3000 for wn in wavenumbers):
            analysis_parts.append("- **高分子材料の可能性**: アルキル基C-H伸縮が検出")
        
        analysis_parts.append("\n**注意**: この解析は一般的なピーク帰属に基づく推定です。正確な同定には追加の分析手法との組み合わせが必要です。")
        
        return "\n".join(analysis_parts)

# 既存の関数群（変更なし）
def create_features_labels(spectra, window_size=10):
    # 特徴量とラベルの配列を初期化
    X = []
    y = []
    # スペクトルデータの長さ
    n_points = len(spectra)
    # 人手によるピークラベル、または自動生成コードをここに配置
    peak_labels = np.zeros(n_points)

    # 特徴量とラベルの抽出
    for i in range(window_size, n_points - window_size):
        # 前後の窓サイズのデータを特徴量として使用
        features = spectra[i-window_size:i+window_size+1]
        X.append(features)
        y.append(peak_labels[i])

    return np.array(X), np.array(y)

def find_index(rs_array,  rs_focused):
    '''
    Convert the index of the proximate wavenumber by finding the absolute 
    minimum value of (rs_array - rs_focused)
    
    input
        rs_array: Raman wavenumber
        rs_focused: Index
    output
        index
    '''

    diff = [abs(element - rs_focused) for element in rs_array]
    index = np.argmin(diff)
    return index

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.array(x, dtype=np.float64)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T).toarray().flatten()
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, dssn_th, lambda_, porder, itermax):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram or spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is, the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    # マイナス値がある場合の処理
    min_value = np.min(x)
    offset = 0
    if min_value < 0:
        offset = abs(min_value) + 1  # 最小値を1にするためのオフセット
        x = x + offset  # 全体をシフト
    
    m = x.shape[0]
    w = np.ones(m, dtype=np.float64)  # 明示的に型を指定
    x = np.asarray(x, dtype=np.float64)  # xも明示的に型を指定
    
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        
        # dssn がゼロまたは非常に小さい場合を回避
        if dssn < 1e-10:
            dssn = 1e-10
        
        # 収束判定
        if (dssn < dssn_th * (np.abs(x)).sum()) or (i == itermax):
            if i == itermax:
                print('WARNING: max iteration reached!')
            break
        
        # 重みの更新
        w[d >= 0] = 0  # d > 0 はピークの一部として重みを無視
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        
        # 境界条件の調整
        if d[d < 0].size > 0:
            w[0] = np.exp(i * np.abs(d[d < 0]).max() / dssn)
        else:
            w[0] = 1.0  # 適切な初期値
        
        w[-1] = w[0]

    return z

def detect_file_type(data):
    """
    Determine the structure of the input data.
    """
    try:
        if data.columns[0].split(':')[0] == "# Laser Wavelength":
            return "ramaneye_new"
        elif data.columns[0] == "WaveNumber":
            return "ramaneye_old"
        elif data.columns[0] == "Pixels":
            return "eagle"
        elif data.columns[0] == "ENLIGHTEN Version":
            return "wasatch"
        return "unknown"
    except:
        return "unknown"

def read_csv_file(uploaded_file, file_extension):
    """
    Read a CSV or TXT file into a DataFrame based on file extension.
    """
    try:
        uploaded_file.seek(0)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', header=0, index_col=None, on_bad_lines='skip')
        return data
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def calculate_peak_width(spectrum, peak_idx, wavenum):
    """
    ピークの半値幅（FWHM）を計算する関数
    
    Parameters:
    spectrum (ndarray): スペクトルデータ
    peak_idx (int): ピークのインデックス
    wavenum (ndarray): 波数データ
    
    Returns:
    fwhm (float): 半値幅 (cm⁻¹)
    """
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return 0.0
    
    peak_intensity = spectrum[peak_idx]
    half_max = peak_intensity / 2.0
    
    # ピークから左側に向かって半値点を探す
    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_max:
        left_idx -= 1
    
    # 線形補間で正確な半値点を求める
    if left_idx < peak_idx and spectrum[left_idx] <= half_max < spectrum[left_idx + 1]:
        # 線形補間
        ratio = (half_max - spectrum[left_idx]) / (spectrum[left_idx + 1] - spectrum[left_idx])
        left_wavenum = wavenum[left_idx] + ratio * (wavenum[left_idx + 1] - wavenum[left_idx])
    else:
        left_wavenum = wavenum[left_idx] if left_idx >= 0 else wavenum[0]
    
    # ピークから右側に向かって半値点を探す
    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
        right_idx += 1
    
    # 線形補間で正確な半値点を求める
    if right_idx > peak_idx and spectrum[right_idx] <= half_max < spectrum[right_idx - 1]:
        # 線形補間
        ratio = (half_max - spectrum[right_idx]) / (spectrum[right_idx - 1] - spectrum[right_idx])
        right_wavenum = wavenum[right_idx] + ratio * (wavenum[right_idx - 1] - wavenum[right_idx])
    else:
        right_wavenum = wavenum[right_idx] if right_idx < len(wavenum) else wavenum[-1]
    
    # 半値幅を計算
    fwhm = abs(right_wavenum - left_wavenum)
    return fwhm

def remove_outliers_and_interpolate(spectrum, window_size=10, threshold_factor=3):
    """
    スペクトルからスパイク（外れ値）を検出し、補完する関数
    スパイクは、ウィンドウ内の標準偏差が一定の閾値を超える場合に検出される
    
    input:
        spectrum: numpy array, ラマンスペクトル
        window_size: ウィンドウのサイズ（デフォルトは20）
        threshold_factor: 標準偏差の閾値（デフォルトは5倍）
    
    output:
        cleaned_spectrum: numpy array, スパイクを取り除き補完したスペクトル
    """
    spectrum_len = len(spectrum)
    cleaned_spectrum = spectrum.copy()
    
    for i in range(spectrum_len):
        # 端点では、ウィンドウサイズが足りないので、ウィンドウを調整
        left_idx = max(i - window_size, 0)
        right_idx = min(i + window_size + 1, spectrum_len)
        
        # ウィンドウ内のデータを取得
        window = spectrum[left_idx:right_idx]
        
        # ウィンドウ内の中央値と標準偏差を計算
        window_median = np.median(window)
        window_std = np.std(window)
        
        # ウィンドウ内の値が標準偏差の閾値を超えるスパイクを検出
        if abs(spectrum[i] - window_median) > threshold_factor * window_std:
            # スパイクが見つかった場合、その値を両隣の中央値で補完
            if i > 0 and i < spectrum_len - 1:  # 両隣の値が存在する場合
                cleaned_spectrum[i] = (spectrum[i - 1] + spectrum[i + 1]) / 2
            elif i == 0:  # 左端の場合
                cleaned_spectrum[i] = spectrum[i + 1]
            elif i == spectrum_len - 1:  # 右端の場合
                cleaned_spectrum[i] = spectrum[i - 1] 
    return cleaned_spectrum
    
# グリッドサーチ関数
def optimize_thresholds_via_gridsearch(
    wavenum, spectrum, second_derivative,
    manual_add_peaks, manual_exclude_indices,
    current_prom_thres, current_deriv_thres,
    detected_original_peaks=None,
    resolution=40
):
    best_score = -np.inf
    best_prom_thres = current_prom_thres
    best_deriv_thres = current_deriv_thres

    # 倍半分の範囲でログスケール検索
    prom_range = np.linspace(current_prom_thres / 2, current_prom_thres * 2, resolution)
    deriv_range = np.linspace(current_deriv_thres / 2, current_deriv_thres * 2, resolution)

    for prom_thres in prom_range:
        for deriv_thres in deriv_range:
            peaks, _ = find_peaks(-second_derivative, height=deriv_thres)
            prominences = peak_prominences(-second_derivative, peaks)[0]
            mask = prominences > prom_thres
            final_peaks = set(peaks[mask])

            score = 0

            # 手動追加が含まれていれば +1
            for x, _ in manual_add_peaks:
                idx = np.argmin(np.abs(wavenum - x))
                if idx in final_peaks:
                    score += 1

            # 除外対象が含まれていれば -1
            for idx in manual_exclude_indices:
                if idx in final_peaks:
                    score -= 1

            # 自動ピークの逸脱ペナルティ
            if len(detected_original_peaks) > 0:
                for idx in final_peaks:
                    if idx not in detected_original_peaks:
                        score -= 1  # 新たに現れた余分なピーク
                for idx in detected_original_peaks:
                    if idx not in final_peaks:
                        score -= 1  # 元のピークが消えた

            # スコア最大のパラメータを保存
            if score > best_score:
                best_score = score
                best_prom_thres = prom_thres
                best_deriv_thres = deriv_thres

    return {
        "prominence_threshold": best_prom_thres,
        "second_deriv_threshold": best_deriv_thres,
        "score": best_score
    }

def spectrum_analysis_mode():
    st.header("📊 ラマンスペクトル解析")
    
   # --- LLM/RAG設定セクション ---
    st.sidebar.subheader("🤖 AI解析設定")
    
    # ✅ 追加（AI有効/無効切り替え）
    enable_ai = st.sidebar.checkbox(
        "🧠 AI機能を有効にする",
        value=True,
        help="AIによる自動成分推定と考察を実行します。"
    )
    
    # RAG機能の有効/無効を選択
    enable_rag = st.sidebar.checkbox(
        "📚 RAG機能を有効にする",
        value=True,
        help="論文データベースからの情報検索機能。無効にすると軽量化されます。"
    )
    
    # Mistralモデル選択
    model_options = [
        "cl-tohoku/bert-base-japanese",
        "microsoft/DialoGPT-medium",  # より軽量な代替モデル
    ]
    
    selected_model = st.sidebar.selectbox(
        "🧠 使用するモデル",
        model_options,
        index=0,
        help="使用するHugging Faceモデルを選択してください。上位ほど軽量です。"
    )
    
    # RAG機能が有効な場合のみファイルアップロード表示
    uploaded_files = []
    if enable_rag:
        uploaded_files = st.sidebar.file_uploader(
            "📄 文献PDFを選択してください（複数可）",
            type=["pdf"],
            accept_multiple_files=True
        )
    
    # 一時保存用ディレクトリ
    TEMP_DIR = "./tmp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # RAGシステムの初期化（必要時のみ）
    if enable_rag:
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = SimpleRAGSystem()
            st.session_state.rag_db_built = False
        
        # ファイルを保存し、ベクトルDBを構築
        if st.sidebar.button("📚 論文データベース構築"):
            if not uploaded_files:
                st.sidebar.warning("文献ファイルを選択してください。")
            else:
                with st.spinner("論文をアップロードし、データベースを構築中..."):
                    # アップロードされたファイルを一時保存
                    for uploaded_file in uploaded_files:
                        save_path = os.path.join(TEMP_DIR, uploaded_file.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # 保存したフォルダを使ってベクトルDB構築
                    st.session_state.rag_system.build_vector_database(TEMP_DIR)
                    st.session_state.rag_db_built = True
                    st.sidebar.success(f"✅ {len(uploaded_files)} 件のPDFからデータベースを構築しました。")
        
        # データベース状態表示
        if st.session_state.rag_db_built:
            st.sidebar.success("✅ 論文データベース構築済み")
        else:
            st.sidebar.info("ℹ️ 論文データベース未構築")
    else:
        # RAG無効時の状態初期化
        if 'rag_system' in st.session_state:
            del st.session_state.rag_system
        st.session_state.rag_db_built = False
        st.sidebar.info("💨 軽量モード（RAG機能無効）")
    
    # LLMの初期化（AI機能有効時のみ）
    if enable_ai:
        if 'simple_llm' not in st.session_state:
            st.session_state.simple_llm = None
            st.session_state.current_model = None
            
        # モデル変更時の再初期化
        if st.session_state.current_model != selected_model:
            st.session_state.simple_llm = None
            st.session_state.current_model = selected_model
        
        st.sidebar.success("🤖 AI解析機能有効")
    else:
        # AI無効時の状態初期化
        if 'simple_llm' in st.session_state:
            del st.session_state.simple_llm
        st.session_state.current_model = None
        st.sidebar.info("💨 軽量モード（AI機能無効）")
    
    # サイドバーに補足指示欄を追加
    user_hint = st.sidebar.text_area(
        "🧪 AIへの補足ヒント（任意）",
        placeholder="例：この試料はポリエチレン系高分子である可能性がある、など"
    )
    
    st.subheader("ピーク検出結果")

    # --- 事前パラメータ ---
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    
    # --- セッションステートの初期化（UI表示よりも前！） ---
    for key, default in {
        "prominence_threshold": 100,
        "second_deriv_threshold": 100,
        "savgol_wsize": 5,
        "spectrum_type_select": "ベースライン削除",
        "second_deriv_smooth": 5,
        "manual_peak_keys": []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    # --- UIパネル（Sidebar） ---
    start_wavenum = st.sidebar.number_input("波数（開始）:", -200, 4800, value=pre_start_wavenum, step=100)
    end_wavenum = st.sidebar.number_input("波数（終了）:", -200, 4800, value=pre_end_wavenum, step=100)
    dssn_th = st.sidebar.number_input("ベースラインパラメーター:", 1, 10000, value=1000, step=1) / 1e7
    
    savgol_wsize = st.sidebar.number_input(
        "ウィンドウサイズ:", 3, 101, 
        value=st.session_state["savgol_wsize"], step=2, key="savgol_wsize"
    )
    
    st.sidebar.subheader("ピーク検出設定")
    
    spectrum_type = st.sidebar.selectbox(
        "解析スペクトル:", ["ベースライン削除", "移動平均後"], 
        index=0, key="spectrum_type_select"
    )
    
    second_deriv_smooth = st.sidebar.number_input(
        "2次微分平滑化:", 3, 35,
        value=st.session_state["second_deriv_smooth"],
        step=2, key="second_deriv_smooth"
    )
    
    # 一時的なセッション変数が存在するならそちらを使う（なければ通常のセッション値）
    prom_default = float(st.session_state.get("prominence_threshold_temp", st.session_state.get("prominence_threshold", 100.0)))
    second_default = float(st.session_state.get("second_deriv_threshold_temp", st.session_state.get("second_deriv_threshold", 100.0)))
    
    second_deriv_threshold = st.sidebar.number_input(
        "2次微分閾値:",
        min_value=0.0,
        max_value=1000.0,
        value=second_default,
        step=10.0,
        key="second_deriv_threshold"
    )
    
    peak_prominence_threshold = st.sidebar.number_input(
        "ピークProminence閾値:",
        min_value=0.0,
        max_value=1000.0,
        value=prom_default,
        step=10.0,
        key="prominence_threshold"
    )

    # --- ファイルアップロード（1箇所に統一） ---
    uploaded_spectrum_files = st.file_uploader("スペクトルファイルを選択してください", accept_multiple_files=True, key="spectrum_file_uploader")
    
    # --- アップロードファイル変更検出 ---
    new_filenames = [f.name for f in uploaded_spectrum_files] if uploaded_spectrum_files else []
    prev_filenames = st.session_state.get("uploaded_filenames", [])

    # --- 設定変更検出 ---
    config_keys = ["spectrum_type_select", "second_deriv_smooth", "second_deriv_threshold", "prominence_threshold"]
    config_changed = any(
        st.session_state.get(f"prev_{key}") != st.session_state[key] for key in config_keys
    )
    file_changed = new_filenames != prev_filenames

    # --- 手動ピーク初期化条件 ---
    if config_changed or file_changed:
        for key in list(st.session_state.keys()):
            if key.endswith("_manual_peaks"):
                del st.session_state[key]
        st.session_state["manual_peak_keys"] = []
        st.session_state["uploaded_filenames"] = new_filenames
        for k in config_keys:
            st.session_state[f"prev_{k}"] = st.session_state[k]
            
    file_labels = []
    all_spectra = []
    all_bsremoval_spectra = []
    all_averemoval_spectra = []
    all_wavenum = []
    
    if uploaded_spectrum_files:
        config_keys = [
            "spectrum_type_select",
            "second_deriv_smooth",
            "second_deriv_threshold",
            "prominence_threshold"
        ]
        # セーフな代入処理（KeyError防止）
        for k in config_keys:
            st.session_state[f"prev_{k}"] = st.session_state.get(k)

        
        # --- ファイル変更検出 ---
        file_changed = new_filenames != prev_filenames
        
        # --- 手動ピーク初期化条件 ---
        if config_changed or file_changed:
            for key in list(st.session_state.keys()):
                if key.endswith("_manual_peaks"):
                    del st.session_state[key]
            st.session_state["manual_peak_keys"] = []
            st.session_state["uploaded_filenames"] = new_filenames
        
            # 安全に prev_ をセット
            for k in config_keys:
                st.session_state[f"prev_{k}"] = st.session_state.get(k)
                
        for uploaded_file in uploaded_spectrum_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1] if '.' in file_name else ''

            try:
                data = read_csv_file(uploaded_file, file_extension)
                file_type = detect_file_type(data)
                uploaded_file.seek(0)
                if file_type == "unknown":
                    st.error(f"{file_name}のファイルタイプを判別できません。")
                    continue

                # 各ファイルタイプに対する処理
                if file_type == "wasatch":
                    st.write(f"ファイルタイプ: Wasatch ENLIGHTEN - {file_name}")
                    lambda_ex = 785
                    data = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=46)
                    pre_wavelength = np.array(data["Wavelength"].values)
                    pre_wavenum = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
                    pre_spectra = np.array(data["Processed"].values)

                elif file_type == "ramaneye_old":
                    st.write(f"ファイルタイプ: RamanEye Data - {file_name}")
                    pre_wavenum = data["WaveNumber"]
                    pre_spectra = np.array(data.iloc[:, -1])
                    if pre_wavenum.iloc[0] > pre_wavenum.iloc[1]:
                        # pre_wavenum と pre_spectra を反転
                        pre_wavenum = pre_wavenum[::-1]
                        pre_spectra = pre_spectra[::-1]
                        
                elif file_type == "ramaneye_new":
                    st.write(f"ファイルタイプ: RamanEye Data - {file_name}")
                    
                    data = pd.read_csv(uploaded_file, skiprows=9)
                    pre_wavenum = data["WaveNumber"]
                    pre_spectra = np.array(data.iloc[:, -1])

                    if pre_wavenum.iloc[0] > pre_wavenum.iloc[1]:
                        # pre_wavenum と pre_spectra を反転
                        pre_wavenum = pre_wavenum[::-1]
                        pre_spectra = pre_spectra[::-1]
                        
                elif file_type == "eagle":
                    st.write(f"ファイルタイプ: Eagle Data - {file_name}")
                    data_transposed = data.transpose()
                    header = data_transposed.iloc[:3]  # 最初の3行
                    reversed_data = data_transposed.iloc[3:].iloc[::-1]
                    data_transposed = pd.concat([header, reversed_data], ignore_index=True)
                    pre_wavenum = np.array(data_transposed.iloc[3:, 0])
                    pre_spectra = np.array(data_transposed.iloc[3:, 1])
                
                start_index = find_index(pre_wavenum, start_wavenum)
                end_index = find_index(pre_wavenum, end_wavenum)

                wavenum = np.array(pre_wavenum[start_index:end_index+1])
                spectra = np.array(pre_spectra[start_index:end_index+1])

                # Baseline and spike removal 
                spectra_spikerm = remove_outliers_and_interpolate(spectra)
                mveAve_spectra = signal.medfilt(spectra_spikerm, savgol_wsize)
                lambda_ = 10e2
                baseline = airPLS(mveAve_spectra, dssn_th, lambda_, 2, 30)
                BSremoval_specta = spectra_spikerm - baseline
                BSremoval_specta_pos = BSremoval_specta + abs(np.minimum(spectra_spikerm, 0))  # 負値を補正

                # 移動平均後のスペクトル
                Averemoval_specta = mveAve_spectra  - baseline
                Averemoval_specta_pos = Averemoval_specta + abs(np.minimum(mveAve_spectra, 0))  # 負値を補正

                # 各スペクトルを格納
                file_labels.append(file_name)  # ファイル名を追加
                all_wavenum.append(wavenum)
                all_spectra.append(spectra)
                all_bsremoval_spectra.append(BSremoval_specta_pos)
                all_averemoval_spectra.append(Averemoval_specta_pos)
                
            except Exception as e:
                st.error(f"{file_name}の処理中にエラーが発生しました: {e}")
        
        # ピーク検出の実行
        if 'peak_detection_triggered' not in st.session_state:
            st.session_state['peak_detection_triggered'] = False
    
        # ボタンを押したらトリガーをONに
        if st.button("ピーク検出を実行"):
            st.session_state['peak_detection_triggered'] = True
        
        if st.session_state['peak_detection_triggered']:
            st.subheader("ピーク検出結果")
            
            peak_results = []
            
            # 現在の設定を表示
            st.info(f"""
            **検出設定:**
            - スペクトルタイプ: {spectrum_type}
            - 2次微分平滑化: {second_deriv_smooth}, 閾値: {second_deriv_threshold} (ピーク検出用)
            - ピーク卓立度閾値: {peak_prominence_threshold}
            """)
            
            for i, file_name in enumerate(file_labels):
                # 選択されたスペクトルタイプに応じてデータを選択
                if spectrum_type == "ベースライン削除":
                    selected_spectrum = all_bsremoval_spectra[i]
                else:  # 移動平均後
                    selected_spectrum = all_averemoval_spectra[i]
                
                wavenum = all_wavenum[i]
                
                # 2次微分計算（ピーク検出用）
                if len(selected_spectrum) > second_deriv_smooth:
                    second_derivative = savgol_filter(selected_spectrum, second_deriv_smooth, 2, deriv=2)
                else:
                    second_derivative = np.gradient(np.gradient(selected_spectrum))
                
                # 2次微分のみによるピーク検出（prominence判定付き, propertiesはダミー）
                peaks, properties = find_peaks(-second_derivative, height=second_deriv_threshold)
                all_peaks, properties = find_peaks(-second_derivative)

                if len(peaks) > 0:
                    # Peak prominences を計算
                    prominences = peak_prominences(-second_derivative, peaks)[0]
                    all_prominences = peak_prominences(-second_derivative, all_peaks)[0]

                    # Prominence 閾値でフィルタリング
                    mask = prominences > peak_prominence_threshold
                    filtered_peaks = peaks[mask]
                    filtered_prominences = prominences[mask]
                    
                    # ±2ピクセル範囲内で局所極大を再確認・補正
                    corrected_peaks = []
                    corrected_prominences = []
                    
                    for peak_idx, prom in zip(filtered_peaks, filtered_prominences):
                        # ±2の範囲内で最大値を探す（範囲超えないよう制限）
                        window_start = max(0, peak_idx - 2)
                        window_end = min(len(selected_spectrum), peak_idx + 3)
                        local_window = selected_spectrum[window_start:window_end]
                        
                        # 局所最大値の位置（ローカルインデックス）
                        local_max_idx = np.argmax(local_window)
                        corrected_idx = window_start + local_max_idx
                    
                        corrected_peaks.append(corrected_idx)
                        
                        # Prominence も再計算
                        local_prom = peak_prominences(-second_derivative, [corrected_idx])[0][0]
                        corrected_prominences.append(local_prom)
                    
                    # numpy配列に変換
                    filtered_peaks = np.array(corrected_peaks)
                    filtered_prominences = np.array(corrected_prominences)
                else:
                    filtered_peaks = np.array([])
                    filtered_prominences = np.array([])
                
                # 結果を保存
                peak_data = {
                    'file_name': file_name,
                    'detected_peaks': filtered_peaks,
                    'detected_prominences': filtered_prominences,
                    'wavenum': wavenum,
                    'spectrum': selected_spectrum,
                    'second_derivative': second_derivative,
                    'second_deriv_smooth': second_deriv_smooth,
                    'second_deriv_threshold': second_deriv_threshold,
                    'prominence_threshold': peak_prominence_threshold,
                    'all_peaks': all_peaks,
                    'all_prominences': all_prominences,
                }
                peak_results.append(peak_data)
                
                # 結果を表示
                st.write(f"**{file_name}**")
                st.write(f"検出されたピーク数: {len(filtered_peaks)} (2次微分 + prominence判定)")
                
                # ピーク情報をテーブルで表示
                if len(filtered_peaks) > 0:
                    peak_wavenums = wavenum[filtered_peaks]
                    peak_intensities = selected_spectrum[filtered_peaks]
                    st.write("**検出されたピーク:**")
                    peak_table = pd.DataFrame({
                        'ピーク番号': range(1, len(peak_wavenums) + 1),
                        '波数 (cm⁻¹)': [f"{wn:.1f}" for wn in peak_wavenums],
                        '強度': [f"{intensity:.3f}" for intensity in peak_intensities],
                        'Prominence': [f"{prom:.4f}" for prom in filtered_prominences]
                    })
                    st.table(peak_table)
                else:
                    st.write("ピークが検出されませんでした")
            
                for result in peak_results:
                    file_key = result['file_name']
                    
                    filtered_peaks = result['detected_peaks']
                    filtered_prominences = result['detected_prominences']
                
                    fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        subplot_titles=[
                            f'{file_key} - {spectrum_type}',
                            f'{file_key} - 微分スペクトル比較',
                            f'{file_key} - Prominence vs 波数'
                        ],
                        vertical_spacing=0.07,
                        row_heights=[0.4, 0.3, 0.3]
                    )
                
                    # 上段: スペクトル表示
                    fig.add_trace(
                        go.Scatter(
                            x=result['wavenum'],
                            y=result['spectrum'],
                            mode='lines',
                            name=spectrum_type,
                            line=dict(color='blue', width=1)
                        ),
                        row=1, col=1
                    )
                
                    if len(filtered_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=result['wavenum'][filtered_peaks],
                                y=result['spectrum'][filtered_peaks],
                                mode='markers',
                                name='検出ピーク（有効）',
                                marker=dict(color='red', size=8, symbol='circle')
                            ),
                            row=1, col=1
                        )
                
                    # 中段: 2次微分表示
                    fig.add_trace(
                        go.Scatter(
                            x=result['wavenum'],
                            y=result['second_derivative'],
                            mode='lines',
                            name='2次微分',
                            line=dict(color='purple', width=1)
                        ),
                        row=2, col=1
                    )
                
                    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
                
                    # 下段: Prominenceプロット
                    fig.add_trace(
                        go.Scatter(
                            x=result['wavenum'][result['all_peaks']],
                            y=result['all_prominences'],
                            mode='markers',
                            name='全ピークのProminence',
                            marker=dict(color='orange', size=4)
                        ),
                        row=3, col=1
                    )
                
                    if len(filtered_peaks) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=result['wavenum'][filtered_peaks],
                                y=filtered_prominences,
                                mode='markers',
                                name='有効なProminence',
                                marker=dict(color='red', size=7, symbol='circle')
                            ),
                            row=3, col=1
                        )
                
                    fig.update_layout(height=800, margin=dict(t=80, b=40))
                    fig.update_xaxes(title_text="波数 (cm⁻¹)", row=3, col=1)
                    fig.update_yaxes(title_text="強度", row=1, col=1)
                    fig.update_yaxes(title_text="微分値", row=2, col=1)
                
                    # ✅ Cloud互換のため st.plotly_chart を使用
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI解析セクション - ピーク確定後の考察機能
                st.markdown("---")
                st.subheader(f"🤖 AI解析 - {file_key}")
                
                # 最終的なピーク情報を収集（自動検出 + 手動追加 - 除外）
                final_peak_data = []
                
                # 有効な自動検出ピーク
                for idx, prom in zip(filtered_peaks, filtered_prominences):
                    final_peak_data.append({
                        'wavenumber': result['wavenum'][idx],
                        'intensity': result['spectrum'][idx],
                        'prominence': prom,
                        'type': 'auto'
                    })
                
                # 手動追加ピーク
                for x, y in st.session_state[f"{file_key}_manual_peaks"]:
                    idx = np.argmin(np.abs(result['wavenum'] - x))
                    try:
                        prom = peak_prominences(-result['second_derivative'], [idx])[0][0]
                    except:
                        prom = 0.0
                    
                    final_peak_data.append({
                        'wavenumber': x,
                        'intensity': y,
                        'prominence': prom,
                        'type': 'manual'
                    })
                
                if final_peak_data:
                    st.write(f"**最終確定ピーク数: {len(final_peak_data)}**")
                    
                    # ピーク表示
                    peak_summary_df = pd.DataFrame([
                        {
                            'ピーク番号': i+1,
                            '波数 (cm⁻¹)': f"{peak['wavenumber']:.1f}",
                            '強度': f"{peak['intensity']:.3f}",
                            'Prominence': f"{peak['prominence']:.3f}",
                            'タイプ': '自動検出' if peak['type'] == 'auto' else '手動追加'
                        }
                        for i, peak in enumerate(final_peak_data)
                    ])
                    
                    # 基本解析情報の表示（常に表示）
                    st.info("🔬 基本解析情報")
                    st.write("検出されたピークの化学的解釈：")
                    
                    # 基本的なピーク解釈
                    analyzer = RamanSpectrumAnalyzer()
                    basic_analysis = analyzer._generate_basic_analysis(final_peak_data)
                    st.markdown(basic_analysis)
                    
                    # 基本レポートのダウンロード
                    basic_report = f"""ラマンスペクトル基本解析レポート
ファイル名: {file_key}
解析日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

=== 検出ピーク情報 ===
{peak_summary_df.to_string(index=False)}

=== 基本解析 ===
{basic_analysis}
"""
                    st.download_button(
                        label="📄 基本解析レポートをダウンロード",
                        data=basic_report,
                        file_name=f"raman_basic_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"download_basic_report_{file_key}"
                    )
                    
                    # AI解析実行ボタン（AI機能有効時のみ表示）
                    if enable_ai:
                        if st.button(f"🧠 AI解析を実行 - {file_key}", key=f"ai_analysis_{file_key}"):
                            # LLMの初期化（必要時のみ）
                            if st.session_state.simple_llm is None:
                                st.session_state.simple_llm = SimpleLLM(selected_model)
                            
                            with st.spinner("AI言語モデルで解析中です。しばらくお待ちください..."):
                                analysis_report = None
                                start_time = time.time()
                        
                                try:
                                    # 関連文献を検索（RAG有効時のみ）
                                    relevant_docs = []
                                    if enable_rag and hasattr(st.session_state, 'rag_system') and st.session_state.rag_db_built:
                                        search_terms = ' '.join([f"{p['wavenumber']:.0f}cm-1" for p in final_peak_data[:5]])
                                        search_query = f"ラマンスペクトロスコピー ピーク {search_terms}"
                                        relevant_docs = st.session_state.rag_system.search_relevant_documents(search_query, top_k=5)
                        
                                    # AIへのプロンプトを生成
                                    analysis_prompt = analyzer.generate_analysis_prompt(
                                        peak_data=final_peak_data,
                                        relevant_docs=relevant_docs,
                                        user_hint=user_hint
                                    )
                                    
                                    # ストリーム出力用エリア
                                    st.success("✅ AIの応答（リアルタイム表示）")
                                    stream_area = st.empty()
                                    full_response = ""
                        
                                    # AIにストリーム形式で問い合わせ
                                    for chunk in st.session_state.simple_llm.generate_stream_response(analysis_prompt, max_tokens=256):
                                        full_response += chunk
                                        stream_area.markdown(full_response)
                    
                                    # ピーク情報まとめ表
                                    peak_summary_df = pd.DataFrame([
                                        {
                                            'ピーク番号': i + 1,
                                            '波数 (cm⁻¹)': f"{peak['wavenumber']:.1f}",
                                            '強度': f"{peak['intensity']:.3f}",
                                            'Prominence': f"{peak['prominence']:.3f}",
                                            'タイプ': '自動検出' if peak['type'] == 'auto' else '手動追加'
                                        }
                                        for i, peak in enumerate(final_peak_data)
                                    ])
                        
                                    # レポートテキスト生成
                                    analysis_report = f"""ラマンスペクトル解析レポート
ファイル名: {file_key}
解析日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
使用モデル: {selected_model}

=== 検出ピーク情報 ===
{peak_summary_df.to_string(index=False)}

=== AI解析結果 ===
{full_response}

=== 参照文献 ===
"""
                                    for i, doc in enumerate(relevant_docs, 1):
                                        analysis_report += f"{i}. {doc['metadata']['filename']}（類似度: {doc['similarity_score']:.3f}）\n"
                        
                                    # 処理時間の表示
                                    elapsed = time.time() - start_time
                                    st.info(f"🕒 解析にかかった時間: {elapsed:.2f} 秒")
                                    
                                    # 解析結果をセッションに保存
                                    st.session_state[f"{file_key}_ai_analysis"] = {
                                        'analysis': full_response,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'model': selected_model,
                                        'peak_data': final_peak_data
                                    }
                        
                                    # レポートダウンロードボタン
                                    st.download_button(
                                        label="📄 AI解析レポートをダウンロード",
                                        data=analysis_report,
                                        file_name=f"raman_ai_analysis_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        mime="text/plain",
                                        key=f"download_ai_report_{file_key}"
                                    )
                        
                                except Exception as e:
                                    st.error("AI解析中にエラーが発生しました。")
                                    st.code(str(e))
                    
                    # 過去の解析結果表示（AI機能有効時のみ）
                    if enable_ai and f"{file_key}_ai_analysis" in st.session_state:
                        with st.expander("📜 過去のAI解析結果を表示"):
                            past_analysis = st.session_state[f"{file_key}_ai_analysis"]
                            st.write(f"**解析日時:** {past_analysis['timestamp']}")
                            st.write(f"**使用モデル:** {past_analysis['model']}")
                            st.markdown("**解析結果:**")
                            st.markdown(past_analysis['analysis'])
                
                else:
                    st.info("確定されたピークがありません。ピーク検出を実行するか、手動でピークを追加してください。")
                
            # 全ピーク結果をCSVでダウンロード可能にする
            all_peaks_data = []
            for result in peak_results:
                # 検出されたピーク（2次微分 + prominence判定）
                if len(result['detected_peaks']) > 0:
                    peak_wavenums = result['wavenum'][result['detected_peaks']]
                    peak_intensities = result['spectrum'][result['detected_peaks']]
                    for j, (wn, intensity, prominence) in enumerate(zip(peak_wavenums, peak_intensities, result['detected_prominences'])):
                        all_peaks_data.append({
                            'ファイル名': result['file_name'],
                            'ピーク番号': j + 1,
                            '波数_cm-1': f"{wn:.1f}",
                            '強度': f"{intensity:.6f}",
                            'Prominence': f"{prominence:.6f}",
                            'スペクトルタイプ': spectrum_type,
                            '検出方法': '2次微分+prominence',
                            '平滑化数値': result['second_deriv_smooth'],
                            '2次微分閾値': result['second_deriv_threshold'],
                            'Prominence閾値': result['prominence_threshold']
                        })
            
            if all_peaks_data:
                peaks_df = pd.DataFrame(all_peaks_data)
                csv = peaks_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="ピーク検出結果をCSVでダウンロード",
                    data=csv,
                    file_name=f"peak_detection_results_{spectrum_type}_prominence.csv",
                    mime="text/csv"
                )

def main():
    st.set_page_config(
        page_title="AIによるラマンピーク解析", 
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 AIによるラマンピーク解析")
    st.markdown("---")
        
    st.sidebar.markdown("---")
    st.sidebar.header("📋 パラメータ設定")
    
    spectrum_analysis_mode()
    
    # フッター情報
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📚 使用方法
    1. **Mistralモデル選択**: 使用するHugging Face Mistralモデルを選択
    2. **論文アップロード**: RAG機能用の論文PDFをアップロード
    3. **データベース構築**: 論文から検索用データベースを作成
    4. **スペクトルアップロード**: 解析するラマンスペクトルをアップロード
    5. **ピーク検出**: 自動検出 + 手動調整でピークを確定
    6. **AI解析実行**: 確定ピークを基にMistralが考察を生成
    
    ### 🔧 サポートファイル形式
    - **スペクトル**: CSV, TXT (RamanEye, Wasatch, Eagle対応)
    - **論文**: PDF, DOCX, TXT
    
    ### ⚠️ システム要件
    - **GPU推奨**: Mistralモデルは高速化のためGPU使用を推奨
    - **メモリ**: 8GB以上のRAM/VRAMを推奨
    - **依存関係**: requirements.txtの全パッケージが必要
    """)

if __name__ == "__main__":
    main()
