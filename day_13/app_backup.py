"""
Fashion Similarity Search Engine — Streamlit app (online stage).

Consumes the artifacts produced by `day_13_backup.ipynb`:
    artifacts/best_model.pt          fine-tuned ResNet50 (classifier head incl.)
    artifacts/image_embeddings.npy   (N, 2048) L2-normalized embeddings
    artifacts/image_paths.pkl        list[str]  absolute image paths (row-aligned)
    artifacts/metadata.pkl           DataFrame  product metadata (row-aligned)
    artifacts/faiss.index            FAISS IndexFlatIP (cosine on unit vectors)
    artifacts/umap_coords.npy        (N, 2) UMAP coordinates
    artifacts/label_encoder.pkl      sklearn LabelEncoder for the target column
    artifacts/training_history.json  training curves / metrics

Four sections (sidebar):
    1. Similarity Search   — upload an image, retrieve Top-K similar products
    2. Embedding Explorer  — interactive UMAP scatter of the embedding space
    3. Category Explorer   — browse / filter products by category, gender, season
    4. Dataset Analytics   — interactive Plotly charts over the dataset

Run:  streamlit run app_backup.py
Extra deps:  pip install torchvision faiss-cpu
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50

ARTIFACTS = Path("artifacts")

st.set_page_config(page_title="Fashion Similarity Search",
                   page_icon="👗", layout="wide")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
EMB_DIM = 2048

eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_arrays():
    embeddings = np.load(ARTIFACTS / "image_embeddings.npy")
    umap_coords = np.load(ARTIFACTS / "umap_coords.npy")
    with open(ARTIFACTS / "image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
    with open(ARTIFACTS / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f).reset_index(drop=True)
    history = {}
    if (ARTIFACTS / "training_history.json").exists():
        with open(ARTIFACTS / "training_history.json") as f:
            history = json.load(f)
    return embeddings, umap_coords, list(image_paths), metadata, history


@st.cache_resource(show_spinner=False)
def load_index():
    import faiss
    return faiss.read_index(str(ARTIFACTS / "faiss.index"))


@st.cache_resource(show_spinner=False)
def load_encoder():
    """Fine-tuned ResNet50 with the classification head replaced by Identity,
    so a forward pass returns the 2048-d embedding used at indexing time."""
    ckpt = torch.load(ARTIFACTS / "best_model.pt", map_location="cpu")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.fc = nn.Identity()
    model.eval()
    return model, ckpt.get("target_col", "articleType")


@torch.no_grad()
def embed_image(pil_img, encoder):
    x = eval_tf(pil_img.convert("RGB")).unsqueeze(0)
    z = F.normalize(encoder(x), p=2, dim=1)
    return z.numpy().astype("float32")


def try_load_all():
    """Load everything, returning (data, error). error is a message or None."""
    try:
        arrays = load_arrays()
        index = load_index()
        encoder, target_col = load_encoder()
        return (arrays, index, encoder, target_col), None
    except FileNotFoundError as e:
        return None, (f"Missing artifact: `{e.filename}`. Run all cells in "
                      "`day_13_backup.ipynb` first to build the `artifacts/` folder.")
    except ModuleNotFoundError as e:
        return None, (f"Missing package `{e.name}`. Install the app deps:  "
                      "`pip install torchvision faiss-cpu`")


def open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def category_column(metadata, target_col):
    """The best available column to colour / group by."""
    for c in (target_col, "articleType", "subCategory", "masterCategory"):
        if c in metadata.columns:
            return c
    return metadata.columns[0]


# ------------------------------------------------------------------
# Page 1 — Similarity Search
# ------------------------------------------------------------------
def page_similarity_search(arrays, index, encoder, target_col):
    embeddings, umap_coords, image_paths, metadata, history = arrays
    st.header("🔍 Similarity Search")
    st.caption("Upload a fashion image to retrieve visually similar products.")

    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload an image",
                                    type=["jpg", "jpeg", "png", "webp"])
    with col_cfg:
        k = st.slider("Number of results (Top-K)", 4, 24, 12, step=4)

    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    query_img = Image.open(uploaded).convert("RGB")
    st.image(query_img, caption="Query image", width=220)

    q = embed_image(query_img, encoder)
    scores, ids = index.search(q, k)
    scores, ids = scores[0], ids[0]

    st.subheader("Most similar products")
    n_cols = 4
    rows = [ids[i:i + n_cols] for i in range(0, len(ids), n_cols)]
    score_rows = [scores[i:i + n_cols] for i in range(0, len(scores), n_cols)]
    for row, srow in zip(rows, score_rows):
        cols = st.columns(n_cols)
        for c, idx, sc in zip(cols, row, srow):
            with c:
                img = open_image(image_paths[idx])
                if img is not None:
                    c.image(img, use_container_width=True)
                m = metadata.iloc[idx]
                name = m.get("productDisplayName", "—")
                cat = m.get(target_col, m.get("articleType", "—"))
                colour = m.get("baseColour", "—")
                season = m.get("season", "—")
                c.markdown(
                    f"**similarity {float(sc):.3f}**  \n"
                    f"{name}  \n"
                    f"`{cat}` · {colour} · {season}"
                )


# ------------------------------------------------------------------
# Page 2 — Embedding Explorer
# ------------------------------------------------------------------
def page_embedding_explorer(arrays, index, encoder, target_col):
    embeddings, umap_coords, image_paths, metadata, history = arrays
    st.header("🗺️ Embedding Explorer")
    st.caption("2-D UMAP projection of the learned embedding space. "
               "Zoom, pan, filter by category, hover for details, click a point "
               "to inspect the product.")

    cat_col = category_column(metadata, target_col)
    df = metadata.copy()
    df["x"] = umap_coords[:, 0]
    df["y"] = umap_coords[:, 1]
    df["row"] = np.arange(len(df))

    all_cats = sorted(df[cat_col].dropna().unique().tolist())
    default = all_cats[:12] if len(all_cats) > 12 else all_cats
    chosen = st.multiselect(f"Filter by {cat_col}", all_cats, default=default)
    max_pts = st.slider("Max points to plot", 1000, min(30000, len(df)),
                        min(8000, len(df)), step=1000)

    view = df[df[cat_col].isin(chosen)] if chosen else df
    if len(view) > max_pts:
        view = view.sample(max_pts, random_state=0)

    hover = [c for c in ["productDisplayName", "articleType", "baseColour",
                         "season", "gender"] if c in view.columns]
    fig = px.scatter(
        view, x="x", y="y", color=cat_col, custom_data=["row"],
        hover_data=hover, height=650, opacity=0.7,
        title=f"UMAP of {len(view):,} products (coloured by {cat_col})")
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend=dict(itemsizing="constant"))

    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="umap")

    sel = event.get("selection", {}) if isinstance(event, dict) else {}
    points = sel.get("points", []) if isinstance(sel, dict) else []
    if points:
        st.subheader("Selected products")
        cols = st.columns(min(4, len(points)))
        for c, p in zip(cols, points[:4]):
            row = int(p["customdata"][0])
            img = open_image(image_paths[row])
            if img is not None:
                c.image(img, use_container_width=True)
            m = metadata.iloc[row]
            c.markdown(
                f"**{m.get('productDisplayName', '—')}**  \n"
                f"`{m.get(cat_col, '—')}`  \n"
                f"{m.get('baseColour', '—')} · {m.get('season', '—')} · "
                f"{m.get('gender', '—')}"
            )
    else:
        st.info("Box- or lasso-select points on the plot to inspect them.")


# ------------------------------------------------------------------
# Page 3 — Category Explorer
# ------------------------------------------------------------------
def page_category_explorer(arrays, index, encoder, target_col):
    embeddings, umap_coords, image_paths, metadata, history = arrays
    st.header("📂 Category Explorer")
    st.caption("Browse products within a category, filter by gender and season, "
               "and inspect category statistics.")

    cat_col = category_column(metadata, target_col)
    c1, c2, c3 = st.columns(3)
    with c1:
        cats = sorted(metadata[cat_col].dropna().unique().tolist())
        category = st.selectbox(f"{cat_col}", cats)
    with c2:
        genders = ["(all)"] + (sorted(metadata["gender"].dropna().unique().tolist())
                               if "gender" in metadata.columns else [])
        gender = st.selectbox("gender", genders)
    with c3:
        seasons = ["(all)"] + (sorted(metadata["season"].dropna().unique().tolist())
                               if "season" in metadata.columns else [])
        season = st.selectbox("season", seasons)

    mask = metadata[cat_col] == category
    if gender != "(all)" and "gender" in metadata.columns:
        mask &= metadata["gender"] == gender
    if season != "(all)" and "season" in metadata.columns:
        mask &= metadata["season"] == season
    subset = metadata[mask]

    # ---- category statistics ----
    m1, m2, m3 = st.columns(3)
    m1.metric("Products in view", f"{len(subset):,}")
    m2.metric(f"Total in {cat_col}", f"{int((metadata[cat_col] == category).sum()):,}")
    m3.metric("Share of dataset", f"{100 * len(subset) / len(metadata):.1f}%")

    if "baseColour" in subset.columns and len(subset):
        fig = px.bar(subset["baseColour"].value_counts().head(15),
                     title="Colour distribution in view",
                     labels={"value": "count", "index": "colour"})
        st.plotly_chart(fig, use_container_width=True)

    # ---- embedding distribution (where this category sits in UMAP space) ----
    df = metadata.copy()
    df["x"], df["y"] = umap_coords[:, 0], umap_coords[:, 1]
    df["highlight"] = np.where(mask.values, category, "other")
    fig = px.scatter(df.sample(min(8000, len(df)), random_state=0),
                     x="x", y="y", color="highlight",
                     color_discrete_map={"other": "#d9d9d9", category: "#e4572e"},
                     opacity=0.6, height=500, title="Embedding distribution (UMAP)")
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)

    # ---- browse products ----
    st.subheader("Products")
    n_show = st.slider("How many to show", 8, 48, 16, step=8)
    show = subset.head(n_show)
    n_cols = 4
    idxs = show.index.tolist()
    for i in range(0, len(idxs), n_cols):
        cols = st.columns(n_cols)
        for c, row in zip(cols, idxs[i:i + n_cols]):
            img = open_image(image_paths[row])
            if img is not None:
                c.image(img, use_container_width=True)
            m = metadata.iloc[row]
            c.markdown(f"{m.get('productDisplayName', '—')}  \n"
                       f"{m.get('baseColour', '—')} · {m.get('season', '—')}")


# ------------------------------------------------------------------
# Page 4 — Dataset Analytics
# ------------------------------------------------------------------
def page_dataset_analytics(arrays, index, encoder, target_col):
    embeddings, umap_coords, image_paths, metadata, history = arrays
    st.header("📊 Dataset Analytics")

    cat_col = category_column(metadata, target_col)
    a, b, c, d = st.columns(4)
    a.metric("Total images", f"{len(metadata):,}")
    b.metric(f"Classes ({cat_col})", f"{metadata[cat_col].nunique():,}")
    if "masterCategory" in metadata.columns:
        c.metric("Master categories", f"{metadata['masterCategory'].nunique():,}")
    d.metric("Embedding dim", f"{embeddings.shape[1]}")

    # ---- top article types / category histogram ----
    top = metadata[cat_col].value_counts().head(20)
    fig = px.bar(top, title=f"Top {cat_col} (images per class)",
                 labels={"value": "count", "index": cat_col})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    if "gender" in metadata.columns:
        with col1:
            fig = px.pie(metadata, names="gender", title="Gender distribution",
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    if "season" in metadata.columns:
        with col2:
            fig = px.bar(metadata["season"].value_counts(),
                         title="Season distribution",
                         labels={"value": "count", "index": "season"})
            st.plotly_chart(fig, use_container_width=True)

    if "masterCategory" in metadata.columns:
        fig = px.bar(metadata["masterCategory"].value_counts(),
                     title="Master category distribution",
                     labels={"value": "count", "index": "masterCategory"})
        st.plotly_chart(fig, use_container_width=True)

    # ---- images-per-class distribution ----
    counts = metadata[cat_col].value_counts()
    fig = px.histogram(counts, nbins=40,
                       title="Images-per-class distribution",
                       labels={"value": "images in class"})
    st.plotly_chart(fig, use_container_width=True)

    # ---- training curves, if available ----
    if history:
        st.subheader("Training history")
        hist_df = pd.DataFrame({
            "epoch": range(1, len(history.get("train_loss", [])) + 1),
            "train_loss": history.get("train_loss", []),
            "val_loss": history.get("val_loss", []),
            "val_top1": history.get("val_top1", []),
            "val_top5": history.get("val_top5", []),
        })
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(px.line(hist_df, x="epoch",
                                    y=["train_loss", "val_loss"], title="Loss"),
                            use_container_width=True)
        with cc2:
            st.plotly_chart(px.line(hist_df, x="epoch",
                                    y=["val_top1", "val_top5"], title="Accuracy"),
                            use_container_width=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    st.sidebar.title("👗 Fashion Search")
    page = st.sidebar.radio(
        "Navigate",
        ["Similarity Search", "Embedding Explorer",
         "Category Explorer", "Dataset Analytics"],
    )

    data, err = try_load_all()
    if err:
        st.error(err)
        st.stop()
    arrays, index, encoder, target_col = data
    st.sidebar.success(f"{len(arrays[2]):,} products indexed")
    st.sidebar.caption(f"Classification target: `{target_col}`")

    if page == "Similarity Search":
        page_similarity_search(arrays, index, encoder, target_col)
    elif page == "Embedding Explorer":
        page_embedding_explorer(arrays, index, encoder, target_col)
    elif page == "Category Explorer":
        page_category_explorer(arrays, index, encoder, target_col)
    else:
        page_dataset_analytics(arrays, index, encoder, target_col)


if __name__ == "__main__":
    main()
