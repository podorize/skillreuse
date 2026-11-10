"""Construcao e persistencia de caches esparsos sem vocabulos herdados."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from gitskills_reuse.modeling.preprocess import build_preprocessor, predictor_view
from gitskills_reuse.modeling.train import (
    DEFAULT_FEATURES,
    DEFAULT_SAMPLE,
    DEFAULT_TARGETS,
)

DEFAULT_CACHE_DIR = Path("data/processed/cache_sparse")


def build_and_save_cache(
    sample_path: Path = DEFAULT_SAMPLE,
    features_path: Path = DEFAULT_FEATURES,
    targets_path: Path = DEFAULT_TARGETS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    top_k_text: int = 100,
    text_max_features: int = 10_000,
) -> None:
    """Extrai e persiste todas as matrizes esparsas pre-processadas de treino e teste."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    print("=" * 60, flush=True)
    print("INICIANDO CONSTRUCAO DO CACHE DE MATRIZES ESPARSAS", flush=True)
    print(f"Diretorio de destino: {cache_dir}", flush=True)
    print("=" * 60, flush=True)

    # 1. Carregamento dos dados
    print("\n1. Carregando dados de treino e teste...", flush=True)
    features = pq.read_table(features_path).to_pandas()
    targets = pq.read_table(
        targets_path,
        columns=["content_hash", "reuse_intensity", "split_stratum"],
    ).to_pandas()
    text = pq.read_table(sample_path, columns=["content_hash", "content"]).to_pandas()

    merged = features.merge(targets, on="content_hash", how="inner").merge(
        text, on="content_hash", how="inner"
    )

    train_df = merged[merged["split"] == "train"].copy().reset_index(drop=True)
    test_df = merged[merged["split"] == "test"].copy().reset_index(drop=True)

    y_train_binary = train_df["reused"].astype(int).to_numpy()
    y_test_binary = test_df["reused"].astype(int).to_numpy()
    y_train_int = train_df["reuse_intensity"].astype(float).fillna(-1).astype(int).to_numpy()
    y_test_int = test_df["reuse_intensity"].astype(float).fillna(-1).astype(int).to_numpy()

    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas", flush=True)

    # Salvar targets
    np.save(cache_dir / "y_train_binary.npy", y_train_binary)
    np.save(cache_dir / "y_test_binary.npy", y_test_binary)
    np.save(cache_dir / "y_train_intensity.npy", y_train_int)
    np.save(cache_dir / "y_test_intensity.npy", y_test_int)

    # 2. Matrizes Estruturais Puras (46 features)
    print("\n2. Processando matriz estrutural pura (46 features)...", flush=True)
    st_train_pred = predictor_view(
        train_df, use_text=False, use_structural=True, use_repository=False
    )
    st_test_pred = predictor_view(
        test_df, use_text=False, use_structural=True, use_repository=False
    )
    st_trans = build_preprocessor(
        st_train_pred, use_text=False, use_structural=True, use_repository=False
    )
    st_train_sp = st_trans.fit_transform(st_train_pred)
    st_test_sp = st_trans.transform(st_test_pred)
    st_names = list(st_trans.get_feature_names_out())
    save_npz(cache_dir / "X_train_st.npz", st_train_sp)
    save_npz(cache_dir / "X_test_st.npz", st_test_sp)
    print(f"Salvo X_train_st {st_train_sp.shape} e X_test_st {st_test_sp.shape}", flush=True)

    # 3. Matrizes Estrutura + Repositorio (165 features)
    print("\n3. Processando matriz estrutural + repositorio (165 features)...", flush=True)
    sr_train_pred = predictor_view(
        train_df, use_text=False, use_structural=True, use_repository=True
    )
    sr_test_pred = predictor_view(
        test_df, use_text=False, use_structural=True, use_repository=True
    )
    sr_trans = build_preprocessor(
        sr_train_pred, use_text=False, use_structural=True, use_repository=True
    )
    sr_train_sp = sr_trans.fit_transform(sr_train_pred)
    sr_test_sp = sr_trans.transform(sr_test_pred)
    sr_names = list(sr_trans.get_feature_names_out())
    save_npz(cache_dir / "X_train_sr.npz", sr_train_sp)
    save_npz(cache_dir / "X_test_sr.npz", sr_test_sp)
    print(f"Salvo X_train_sr {sr_train_sp.shape} e X_test_sr {sr_test_sp.shape}", flush=True)

    if text_max_features < 1:
        raise ValueError("text_max_features deve ser positivo")
    if top_k_text < 1 or top_k_text > text_max_features:
        raise ValueError("top_k_text deve estar entre 1 e text_max_features")
    print("\n4. Vetorizando texto e selecionando termos por Chi-2...", flush=True)
    text_candidate_source = "outer-training TF-IDF vocabulary"
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=text_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )

    t_txt = time.monotonic()
    txt_train_raw = vec.fit_transform(train_df["content"])
    n_text_candidate_terms = len(vec.get_feature_names_out())
    text_candidate_sha256 = hashlib.sha256(
        "\n".join(vec.get_feature_names_out()).encode("utf-8")
    ).hexdigest()
    (cache_dir / "text_candidate_terms.json").write_text(
        json.dumps(vec.get_feature_names_out().tolist(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    txt_test_raw = vec.transform(test_df["content"])
    print(f"Vetorizacao textual concluida em {time.monotonic() - t_txt:.1f}s", flush=True)

    # Selecao dos Top-100 termos no treino
    k_sel = min(top_k_text, txt_train_raw.shape[1])
    selector = SelectKBest(score_func=chi2, k=k_sel)
    txt_train_sel = selector.fit_transform(txt_train_raw, y_train_binary)
    txt_test_sel = selector.transform(txt_test_raw)
    vocab = np.array(vec.get_feature_names_out())
    selected_text_names = [f"text__{name}" for name in vocab[selector.get_support()]]

    # 5. Montagem e Persistencia da Matriz Hibrida (265 features)
    X_train_hybrid = hstack([sr_train_sp, txt_train_sel]).tocsr()
    X_test_hybrid = hstack([sr_test_sp, txt_test_sel]).tocsr()
    hybrid_names = sr_names + selected_text_names

    save_npz(cache_dir / "X_train_hybrid.npz", X_train_hybrid)
    save_npz(cache_dir / "X_test_hybrid.npz", X_test_hybrid)
    print(
        f"Salvo X_train_hybrid {X_train_hybrid.shape} e X_test_hybrid {X_test_hybrid.shape}",
        flush=True,
    )

    # 6. Salvar nomes de atributos e manifesto
    (cache_dir / "hybrid_feature_names.json").write_text(
        json.dumps(hybrid_names, indent=2) + "\n", encoding="utf-8"
    )
    (cache_dir / "st_feature_names.json").write_text(
        json.dumps(st_names, indent=2) + "\n", encoding="utf-8"
    )
    (cache_dir / "sr_feature_names.json").write_text(
        json.dumps(sr_names, indent=2) + "\n", encoding="utf-8"
    )
    for split_name, split_df in (("train", train_df), ("test", test_df)):
        split_df[["content_hash", "representative_repo_id"]].to_csv(
            cache_dir / f"{split_name}_identifiers.csv", index=False
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "n_st_features": st_train_sp.shape[1],
        "n_sr_features": sr_train_sp.shape[1],
        "n_hybrid_features": X_train_hybrid.shape[1],
        "text_candidate_mode": "outer_training_tfidf",
        "text_candidate_source": text_candidate_source,
        "n_text_candidate_terms": n_text_candidate_terms,
        "text_candidate_sha256": text_candidate_sha256,
        "text_vocab_sha256": text_candidate_sha256,
        "tfidf_parameters": {
            "ngram_range": [1, 2],
            "min_df": 5,
            "max_df": 0.95,
            "max_features": text_max_features,
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "dtype": "float32",
        },
        "text_selection": f"top-{k_sel} chi-square terms selected using outer-training labels only",
        "selected_text_terms_file": (
            "text_candidate_terms.json (full candidate list); "
            "hybrid_feature_names.json (selected features)"
        ),
        "total_elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\n" + "=" * 60, flush=True)
    print(f"CACHE CONSTRUIDO COM SUCESSO EM {time.monotonic() - t0:.1f}s!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    build_and_save_cache()
