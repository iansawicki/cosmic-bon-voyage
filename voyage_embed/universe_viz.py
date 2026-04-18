"""t-SNE + KMeans universe views: Plotly, matplotlib, D3, and sample cache I/O."""

from __future__ import annotations

import json
import html as html_module
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _words_from_metadata(row: dict[str, Any]) -> list[str]:
    import re

    tag_word = re.compile(r"[a-zA-Z][a-zA-Z\-']{2,}")
    stop = frozenset(
        """
        the and for with from that this were been into than then them their what when
        which while will your also only over such some song track music audio feat
        remaster mix edit live version album single ep vol various artists original
        """.split()
    )
    blob = " ".join(
        str(row.get(k) or "")
        for k in (
            "combined_tags",
            "track_title",
            "title",
            "artist_name",
            "artist",
            "album_name",
            "album",
        )
    )
    out = []
    for m in tag_word.finditer(blob.lower()):
        w = m.group(0).strip("'")
        if len(w) > 2 and w not in stop:
            out.append(w)
    return out


def _hover_html_for_row(row: dict[str, Any]) -> str:
    title = html_module.escape(str(row.get("track_title") or row.get("title") or "")[:120])
    artist = html_module.escape(str(row.get("artist_name") or row.get("artist") or "")[:100])
    tags = html_module.escape(str(row.get("combined_tags") or "")[:240])
    tid = html_module.escape(str(row.get("track_id") or row.get("id") or "")[:48])
    return (
        f"<b>{title}</b><br>{artist}<br>"
        f"<span style='font-size:11px'>{tags}</span><br>"
        f"<i>{tid}</i>"
    )


@dataclass
class UniverseModel:
    column: str
    k_target: int
    xy: np.ndarray
    labels: np.ndarray
    meta_rows: list[dict[str, Any]]


def cluster_hints(labels: np.ndarray, meta_rows: list[dict[str, Any]], top_n: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in sorted(int(x) for x in np.unique(labels)):
        mask = labels == k
        idx = np.where(mask)[0]
        c: Counter[str] = Counter()
        for i in idx:
            for w in _words_from_metadata(meta_rows[i]):
                c[w] += 1
        terms = [t for t, _ in c.most_common(top_n)]
        out.append({"id": k, "n": len(idx), "terms": terms})
    return out


def _as_float_vector(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        v = val.astype(np.float64, copy=False).ravel()
        return v if v.size else None
    if isinstance(val, (list, tuple)):
        try:
            v = np.asarray(val, dtype=np.float64).ravel()
            return v if v.size else None
        except (ValueError, TypeError):
            return None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = np.array(json.loads(s), dtype=np.float64).ravel()
                return v if v.size else None
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def compute_universe(
    sample: list[dict[str, Any]],
    embedding_column: str | None,
    max_points: int,
    num_clusters: int,
    guess_embedding_columns: Callable[[list[str]], list[str]],
) -> UniverseModel | None:
    """guess_embedding_columns maps sorted column names to embedding-like candidates."""
    if not sample:
        return None
    columns_sorted = sorted({k for row in sample for k in row.keys()})
    cols = guess_embedding_columns(columns_sorted)
    col = embedding_column or (cols[0] if cols else None)
    if not col:
        return None

    vecs: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for row in sample[:max_points]:
        v = _as_float_vector(row.get(col))
        if v is not None:
            vecs.append(v)
            meta_rows.append(row)

    if len(vecs) < 5:
        return None

    try:
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
    except ImportError as e:
        raise SystemExit("Plot requires sklearn: pip install scikit-learn") from e

    X = np.stack(vecs, axis=0).astype(np.float32)
    n = X.shape[0]
    perplexity = max(5, min(30, n - 1))
    xy = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(X)

    if num_clusters and num_clusters > 1:
        k_target = min(num_clusters, n - 1)
    else:
        k_target = max(5, min(24, max(2, n // 80)))
    k_target = min(k_target, n - 1)
    k_target = max(2, k_target)
    labels = KMeans(n_clusters=k_target, random_state=42, n_init=10).fit_predict(X)

    return UniverseModel(
        column=col,
        k_target=k_target,
        xy=xy,
        labels=labels,
        meta_rows=meta_rows,
    )


def write_plotly_universe(
    out_path: str | Path,
    model: UniverseModel,
    hover_texts: list[str],
    *,
    similarities: np.ndarray | None = None,
    search_query: str = "",
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.colors import qualitative
    except ImportError as e:
        raise SystemExit("Interactive plot requires plotly: pip install plotly>=5.18") from e

    xy = model.xy
    labels = model.labels
    col = model.column
    k_target = model.k_target
    n = xy.shape[0]

    if similarities is not None and len(similarities) == n:
        sims = np.asarray(similarities, dtype=np.float64)
        texts = [
            f"{h}<br><i>cosine sim: {float(sims[i]):.4f}</i>"
            for i, h in enumerate(hover_texts)
        ]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker=dict(
                    size=9,
                    color=sims,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="cosine"),
                    line=dict(width=0.4, color="rgba(255,255,255,0.35)"),
                ),
                text=np.array(texts, dtype=object),
                hovertemplate="%{text}<extra></extra>",
            )
        )
        qdisp = (search_query[:120] + "…") if len(search_query) > 120 else search_query
        fig.update_layout(
            title=f"t-SNE — {col} (n={n}, k={k_target}) · query: {qdisp}",
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            xaxis=dict(title="t-SNE 1", gridcolor="#30363d", zeroline=False),
            yaxis=dict(title="t-SNE 2", gridcolor="#30363d", zeroline=False),
            margin=dict(l=50, r=30, t=60, b=50),
            hovermode="closest",
            showlegend=False,
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn", config={"scrollZoom": True})
        return

    palette = (qualitative.Dark24 + qualitative.Light24) * 3
    fig = go.Figure()
    for k in range(int(labels.max()) + 1):
        mask = labels == k
        if not np.any(mask):
            continue
        n_k = int(np.sum(mask))
        fig.add_trace(
            go.Scatter(
                x=xy[mask, 0],
                y=xy[mask, 1],
                mode="markers",
                name=f"Cluster {k} (n={n_k})",
                marker=dict(
                    size=9,
                    color=palette[k % len(palette)],
                    line=dict(width=0.4, color="rgba(255,255,255,0.35)"),
                ),
                text=np.array(hover_texts, dtype=object)[mask],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"t-SNE — {col} (n={n}, k={k_target} clusters)",
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        legend=dict(
            title="KMeans (embedding space)",
            bgcolor="rgba(22,27,34,0.92)",
            bordercolor="#30363d",
        ),
        xaxis=dict(title="t-SNE 1", gridcolor="#30363d", zeroline=False),
        yaxis=dict(title="t-SNE 2", gridcolor="#30363d", zeroline=False),
        margin=dict(l=50, r=30, t=60, b=50),
        hovermode="closest",
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", config={"scrollZoom": True})


def write_matplotlib_universe(out_path: str | Path, model: UniverseModel) -> None:
    import matplotlib.pyplot as plt

    xy = model.xy
    labels = model.labels
    col = model.column
    k_target = model.k_target
    n = xy.shape[0]
    nlab = int(labels.max()) + 1

    plt.figure(figsize=(11, 9))
    for k in range(nlab):
        mask = labels == k
        if not np.any(mask):
            continue
        plt.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=10,
            alpha=0.75,
            color=plt.cm.tab20((k % 20) / 19.0),
            label=f"C{k}",
            edgecolors="none",
        )
    plt.title(f"t-SNE — {col} (n={n}, k={k_target} clusters)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def _d3_payload(
    model: UniverseModel,
    *,
    search_query: str = "",
    similarities: np.ndarray | None = None,
    dynamic_search: bool = False,
    api_search_path: str = "/api/search",
    stream_proxy: bool = False,
    stream_api_path: str = "/api/sign-audio",
    stream_media_path: str = "/api/stream-audio",
) -> dict[str, Any]:
    hints = cluster_hints(model.labels, model.meta_rows)
    points = []
    sim_mode = similarities is not None and len(similarities) == len(model.meta_rows)
    sim_arr = np.asarray(similarities) if similarities is not None else None
    for i, row in enumerate(model.meta_rows):
        p: dict[str, Any] = {
            "x": float(model.xy[i, 0]),
            "y": float(model.xy[i, 1]),
            "c": int(model.labels[i]),
            "title": str(row.get("track_title") or row.get("title") or "")[:200],
            "artist": str(row.get("artist_name") or row.get("artist") or "")[:160],
            "tags": str(row.get("combined_tags") or "")[:400],
            "id": str(row.get("track_id") or row.get("id") or "")[:64],
        }
        ak = str(row.get("audio_key") or "").strip()
        if ak:
            p["audioKey"] = ak[:512]
        if sim_mode and sim_arr is not None:
            p["sim"] = float(sim_arr[i])
        points.append(p)
    base_title = f"t-SNE — {model.column} (n={len(points)}, k={model.k_target} clusters)"
    if search_query and sim_mode:
        qdisp = search_query[:100] + ("…" if len(search_query) > 100 else "")
        base_title = f"{base_title} · query: {qdisp}"
    out: dict[str, Any] = {
        "title": base_title,
        "embeddingColumn": model.column,
        "kClusters": model.k_target,
        "points": points,
        "clusterHints": hints,
        "simMode": sim_mode,
        "searchQuery": search_query if sim_mode else "",
        "dynamicSearch": bool(dynamic_search),
        "apiSearchPath": api_search_path,
        "streamProxy": bool(stream_proxy),
        "streamApiPath": stream_api_path,
        "streamMediaPath": stream_media_path if stream_proxy else "",
    }
    return out


def build_d3_html_document(
    model: UniverseModel,
    *,
    search_query: str = "",
    similarities: np.ndarray | None = None,
    dynamic_search: bool = False,
    api_search_path: str = "/api/search",
    stream_proxy: bool = False,
    stream_api_path: str = "/api/sign-audio",
    stream_media_path: str = "/api/stream-audio",
) -> str:
    """Render the full D3 HTML string (used by ``write_d3_universe`` and the local cosmos server)."""
    payload = _d3_payload(
        model,
        search_query=search_query,
        similarities=similarities,
        dynamic_search=dynamic_search,
        api_search_path=api_search_path,
        stream_proxy=stream_proxy,
        stream_api_path=stream_api_path,
        stream_media_path=stream_media_path,
    )
    data_json = json.dumps(payload, ensure_ascii=False)
    return _D3_HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


def write_d3_universe(
    out_path: str | Path,
    model: UniverseModel,
    *,
    search_query: str = "",
    similarities: np.ndarray | None = None,
    dynamic_search: bool = False,
    api_search_path: str = "/api/search",
    stream_proxy: bool = False,
    stream_api_path: str = "/api/sign-audio",
    stream_media_path: str = "/api/stream-audio",
) -> None:
    """Write a self-contained HTML file with D3 v7 (zoom, filter, tooltips; optional query similarity coloring)."""
    html_doc = build_d3_html_document(
        model,
        search_query=search_query,
        similarities=similarities,
        dynamic_search=dynamic_search,
        api_search_path=api_search_path,
        stream_proxy=stream_proxy,
        stream_api_path=stream_api_path,
        stream_media_path=stream_media_path,
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_doc, encoding="utf-8")


# Embedded template avoids fetch() for file:// opening
_D3_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Signal field · Bon Voyage</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, sans-serif; background: #0d1117; color: #e6edf3; }
    #wrap { display: flex; flex-direction: column; height: 100vh; }
    header {
      padding: 10px 16px; background: rgba(22,27,34,0.92); border-bottom: 1px solid #30363d;
      display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
      backdrop-filter: blur(8px);
    }
    header h1 { font-size: 1rem; font-weight: 600; margin: 0; flex: 1 1 200px; letter-spacing: 0.02em; }
    #filter { flex: 1 1 220px; min-width: 160px; padding: 8px 10px; border-radius: 6px;
      border: 1px solid #30363d; background: #0d1117; color: #e6edf3; font-size: 14px; }
    #hint { font-size: 12px; color: #8b949e; max-width: 420px; }
    #main { flex: 1; display: flex; min-height: 0; }
    #chart {
      flex: 1; min-width: 0; position: relative; height: 100%;
      background: radial-gradient(ellipse 120% 80% at 50% 100%, rgba(88,166,255,0.06), transparent 55%),
        radial-gradient(ellipse 80% 50% at 20% 30%, rgba(163,113,247,0.05), transparent 45%);
    }
    #starfield {
      position: absolute; inset: 0; pointer-events: none; z-index: 0;
      opacity: 0.55;
      background-image:
        radial-gradient(1.5px 1.5px at 8% 12%, rgba(255,255,255,0.45), transparent),
        radial-gradient(1px 1px at 22% 44%, rgba(200,220,255,0.4), transparent),
        radial-gradient(1px 1px at 91% 18%, rgba(255,255,255,0.35), transparent),
        radial-gradient(1.5px 1.5px at 67% 63%, rgba(255,255,255,0.38), transparent),
        radial-gradient(1px 1px at 41% 88%, rgba(180,200,255,0.32), transparent),
        radial-gradient(1px 1px at 55% 31%, rgba(255,255,255,0.28), transparent),
        radial-gradient(1px 1px at 73% 9%, rgba(255,255,255,0.3), transparent),
        radial-gradient(1.5px 1.5px at 15% 71%, rgba(220,230,255,0.35), transparent),
        radial-gradient(1px 1px at 84% 52%, rgba(255,255,255,0.25), transparent),
        radial-gradient(1px 1px at 33% 26%, rgba(255,255,255,0.22), transparent),
        radial-gradient(1px 1px at 96% 76%, rgba(200,210,255,0.28), transparent),
        radial-gradient(1px 1px at 48% 6%, rgba(255,255,255,0.2), transparent);
      animation: starPulse 18s ease-in-out infinite;
    }
    @keyframes starPulse {
      0%, 100% { opacity: 0.48; }
      50% { opacity: 0.62; }
    }
    #chart > svg { position: relative; z-index: 1; display: block; cursor: grab; }
    #chart > svg:active { cursor: grabbing; }
    #sidebar {
      width: 280px; flex-shrink: 0; overflow-y: auto; padding: 12px;
      background: #161b22; border-left: 1px solid #30363d; font-size: 12px;
    }
    #sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #8b949e; margin: 0 0 8px; }
    .cluster-block { margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #21262d; }
    .cluster-block:last-child { border-bottom: none; }
    .cluster-id { font-weight: 600; color: #58a6ff; }
    .dot { stroke: rgba(255,255,255,.28); }
    .dot.dim { opacity: 0.08; }
    .dot.hl { opacity: 1; stroke: #fff; stroke-width: 1.2; }
    #tooltip {
      position: fixed; pointer-events: none; z-index: 50; max-width: 320px;
      padding: 10px 12px; background: rgba(22,27,34,.95); border: 1px solid #30363d;
      border-radius: 8px; font-size: 12px; line-height: 1.45; display: none;
      box-shadow: 0 8px 24px rgba(0,0,0,.45);
    }
    #tooltip b { color: #f0f6fc; }
    #tooltip .tags { color: #8b949e; font-size: 11px; margin-top: 6px; word-break: break-word; }
    .trip-layer path.trip-glow {
      fill: none; stroke: rgba(163,113,247,0.28); stroke-width: 10; stroke-linecap: round; stroke-linejoin: round;
      pointer-events: none;
    }
    .trip-layer path.trip-core {
      fill: none; stroke: rgba(163,113,247,0.92); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round;
      pointer-events: none;
    }
    .trip-layer circle.trip-token {
      fill: #f0f6fc; stroke: #a371f7; stroke-width: 2px;
      filter: drop-shadow(0 0 8px rgba(163,113,247,0.85));
      pointer-events: none;
    }
    .dot.trip-dim { opacity: 0.1 !important; }
    .dot.trip-stop { opacity: 1 !important; stroke: #a371f7 !important; stroke-width: 2px !important; }
    .dot.flight-now { stroke: #f0f6fc !important; stroke-width: 3px !important; filter: drop-shadow(0 0 10px rgba(163,113,247,0.95)); }
    @keyframes flightPulse {
      0%, 100% { filter: drop-shadow(0 0 6px rgba(163,113,247,0.75)); }
      50% { filter: drop-shadow(0 0 14px rgba(163,113,247,1)); }
    }
    .trip-layer circle.trip-token.flying { animation: flightPulse 1.6s ease-in-out infinite; }
    .link-layer line { stroke: rgba(88,166,255,0.22); stroke-width: 0.65; pointer-events: none; }
    .sector-layer text { fill: rgba(139,148,158,0.88); font-size: 10px; font-weight: 500; letter-spacing: 0.04em;
      text-shadow: 0 0 8px #0d1117, 0 0 2px #0d1117; pointer-events: none; }
    #cosmos-controls label { cursor: pointer; user-select: none; }
    #cosmos-controls input { vertical-align: middle; margin-right: 4px; }
    #signal-readout svg { display: block; margin-top: 6px; border-radius: 4px; }
    #mission-log button { display: block; width: 100%; text-align: left; margin-bottom: 4px; padding: 6px 8px;
      border-radius: 6px; border: 1px solid #30363d; background: #0d1117; color: #8b949e; font-size: 11px; cursor: pointer; }
    #mission-log button:hover { border-color: #58a6ff; color: #e6edf3; }
    #audio-dock {
      display: none;
      position: fixed;
      left: 0; right: 0; bottom: 0;
      z-index: 40;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      background: rgba(22,27,34,0.96);
      border-top: 1px solid #30363d;
      backdrop-filter: blur(8px);
    }
    #audio-dock audio { flex: 0 1 360px; min-width: 180px; max-width: 100%; height: 36px; }
    #audio-dock-meta { flex: 1; min-width: 0; }
    #audio-dock-title { font-size: 13px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  </style>
</head>
<body>
  <div id="wrap">
    <header>
      <h1 id="title">Track universe</h1>
      <div id="cosmos-controls" style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:12px;color:#8b949e;">
        <label title="Nearest neighbors in 2D layout"><input type="checkbox" id="toggle-links"/> Constellation</label>
        <label title="Cluster keyword labels at centroids"><input type="checkbox" id="toggle-sectors" checked/> Sectors</label>
      </div>
      <input id="filter" type="search" placeholder="Filter by title, artist, tags…" autocomplete="off"/>
      <span id="hint"></span>
    </header>
    <div id="main">
      <div id="chart">
        <div id="starfield" aria-hidden="true"></div>
      </div>
      <aside id="sidebar"></aside>
    </div>
  </div>
  <div id="audio-dock" style="display:none" aria-label="Stream player">
    <audio id="stream-audio" controls preload="none"></audio>
    <div id="audio-dock-meta">
      <div id="audio-dock-title">—</div>
      <div id="audio-dock-st" style="font-size:11px;color:#8b949e"></div>
    </div>
  </div>
  <div id="tooltip"></div>
  <script type="application/json" id="embed-data">__DATA_JSON__</script>
  <script>
(function() {
  const DATA = JSON.parse(document.getElementById("embed-data").textContent);
  document.getElementById("title").textContent = DATA.title;
  if (DATA.streamProxy) {
    var dockEl = document.getElementById("audio-dock");
    if (dockEl) dockEl.style.display = "flex";
  }

  const color = d3.scaleOrdinal(d3.schemeTableau10);
  const points = DATA.points;
  let useSim = !!(DATA.simMode && points.length && (points[0].sim !== undefined && points[0].sim !== null));
  let simColor = null;
  let rScale = null;
  function refreshSimColor() {
    if (!useSim) { simColor = null; return; }
    simColor = d3.scaleSequential(d3.interpolateViridis).domain(d3.extent(points, function(d) { return d.sim; }));
  }
  function refreshRScale() {
    if (!useSim || !points.length || points[0].sim === undefined || points[0].sim === null) {
      rScale = null;
      return;
    }
    rScale = d3.scaleLinear().domain(d3.extent(points, function(d) { return d.sim; })).range([3.6, 9.2]);
  }
  refreshSimColor();
  refreshRScale();
  function dotFill(d) {
    if (useSim && d.sim !== undefined && d.sim !== null) return simColor(d.sim);
    return color(d.c);
  }
  function dotRadius(d) {
    if (useSim && rScale && d.sim !== undefined && d.sim !== null) return rScale(d.sim);
    return 5;
  }
  function dotStrokeW(d) {
    if (!useSim || d.sim === undefined || d.sim === null) return 0.38;
    var ext = d3.extent(points, function(p) { return p.sim; });
    var t = (d.sim - ext[0]) / (ext[1] - ext[0] + 1e-12);
    t = Math.max(0, Math.min(1, t));
    return 0.35 + t * 0.95;
  }
  function playSignalPing() {
    try {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      var ctx = new AC();
      var o = ctx.createOscillator();
      var gn = ctx.createGain();
      o.type = "sine";
      o.frequency.setValueAtTime(196, ctx.currentTime);
      o.frequency.exponentialRampToValueAtTime(392, ctx.currentTime + 0.11);
      gn.gain.setValueAtTime(0.0001, ctx.currentTime);
      gn.gain.exponentialRampToValueAtTime(0.065, ctx.currentTime + 0.02);
      gn.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.22);
      o.connect(gn);
      gn.connect(ctx.destination);
      o.start();
      o.stop(ctx.currentTime + 0.24);
      setTimeout(function() { try { ctx.close(); } catch (e) {} }, 400);
    } catch (e) {}
  }
  const tooltip = d3.select("#tooltip");
  let filterText = "";

  const sidebar = d3.select("#sidebar");
  const signalReadout = sidebar.append("div").attr("id", "signal-readout").style("display", "none");

  if (useSim && DATA.searchQuery) {
    sidebar.append("h2").text("Voyage query");
    const qb = sidebar.append("div").attr("class", "cluster-block");
    qb.append("div").style("font-size", "13px").style("line-height", "1.4").text(DATA.searchQuery);
    qb.append("div").style("margin-top", "6px").style("color", "#8b949e").text("Brighter & larger = stronger match (query embedding vs row vector).");
  }
  if (DATA.dynamicSearch) {
    sidebar.append("div").attr("id", "sidebar-voyage").style("display", "none");
    var mlw = sidebar.append("div").attr("id", "mission-log-wrap").style("display", "none");
    mlw.append("h2").text("Mission log");
    mlw.append("div").attr("id", "mission-log");
    sidebar.append("div").attr("id", "tripper-wrap").style("display", "none");
  }
  sidebar.append("h2").text("Cluster hints");
  (DATA.clusterHints || []).forEach(function(ch) {
    const div = sidebar.append("div").attr("class", "cluster-block");
    div.append("div").attr("class", "cluster-id").text("Cluster " + ch.id + " (n=" + ch.n + ")");
    div.append("div").text((ch.terms && ch.terms.length) ? ch.terms.join(", ") : "(no terms)");
  });

  const hintEl = document.getElementById("hint");
  if (DATA.dynamicSearch) {
    hintEl.textContent = "Scan with Voyage in the header · brighter & larger = closer match · zoom · pan · filter";
  } else if (useSim) {
    hintEl.textContent = "Brighter & larger = closer match to query · scroll zoom · drag pan · filter · hover";
  } else {
    hintEl.textContent = "Scroll to zoom · drag to pan · type to filter · hover for details";
  }
  if (DATA.streamProxy) {
    hintEl.textContent = hintEl.textContent + " · Double-click a dot to stream";
  }

  function esc(s) {
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }
  var streamAudio = document.getElementById("stream-audio");
  var audioDockTitle = document.getElementById("audio-dock-title");
  var audioDockSt = document.getElementById("audio-dock-st");
  function bvAudioCanPlay(mime) {
    try {
      var a = document.createElement("audio");
      var r = a.canPlayType(mime);
      return r === "probably" || r === "maybe";
    } catch (e) { return false; }
  }
  function bvUnsupportedFormatHint(audioKey) {
    var k = String(audioKey || "");
    if (/\\.flac$/i.test(k) && !bvAudioCanPlay("audio/flac")) {
      return "FLAC won’t play in this browser (Safari blocks it). Use Chrome, Edge, or Firefox, or double‑click a track whose file ends in .mp3 / .m4a / .aac.";
    }
    if (/\\.aiff?$/i.test(k) && !bvAudioCanPlay("audio/aiff")) {
      return "AIFF isn’t supported in this browser. Try another browser or an MP3.";
    }
    return "";
  }
  if (streamAudio && !streamAudio.dataset.bvErr) {
    streamAudio.dataset.bvErr = "1";
    streamAudio.addEventListener("error", function() {
      var err = streamAudio.error;
      var msg = "Playback failed.";
      if (err) {
        if (err.code === 2) msg = "Network error (proxy or CDN).";
        else if (err.code === 3) msg = "Decode error.";
        else if (err.code === 4) {
          msg = "Codec not supported in this browser (often FLAC in Safari). Try Chrome/Edge, or an MP3/AAC track.";
        }
      }
      if (audioDockSt) audioDockSt.textContent = msg;
      if (typeof flightActive !== "undefined" && flightActive) {
        flightFails = (flightFails || 0) + 1;
        if (flightFails >= FLIGHT_MAX_FAILS) {
          stopGalacticFlight();
          if (audioDockSt) audioDockSt.textContent = "Flight aborted after repeated playback errors.";
        } else {
          scheduleFlightSkip("Error at this stop — skipping…");
        }
      }
    });
    streamAudio.addEventListener("ended", function() {
      if (typeof flightActive !== "undefined" && flightActive) {
        playFlightStop(flightIdx + 1);
      }
    });
    streamAudio.addEventListener("timeupdate", function() {
      if (typeof flightActive !== "undefined" && flightActive) flightTokenTick();
    });
  }
  function playTrackFromDot(d) {
    if (!DATA.streamProxy) return;
    if (!streamAudio || !audioDockTitle || !audioDockSt) return;
    if (flightActive) stopGalacticFlight({ silent: true });
    audioDockTitle.textContent = (d.title || "(no title)") + (d.artist ? (" — " + d.artist) : "");
    if (!d.audioKey) {
      audioDockSt.textContent = "No audio_key for this row.";
      return;
    }
    var hint = bvUnsupportedFormatHint(d.audioKey);
    if (hint) {
      audioDockSt.textContent = hint;
      return;
    }
    var mediaBase = (DATA.streamMediaPath || "/api/stream-audio").replace(/\\/$/, "");
    audioDockSt.textContent = "Loading…";
    function clearLoading() {
      if (audioDockSt) audioDockSt.textContent = "";
    }
    streamAudio.addEventListener("canplay", clearLoading, { once: true });
    streamAudio.src = mediaBase + "?assetName=" + encodeURIComponent(d.audioKey);
    var p = streamAudio.play();
    if (p !== undefined) {
      p.catch(function(err) {
        if (!err) return;
        if (err.name === "AbortError") return;
        if (err.name === "NotAllowedError") {
          audioDockSt.textContent = "Tap ▶ on the audio bar (browser blocked starting playback from this gesture).";
          return;
        }
        audioDockSt.textContent = err.message || "Playback failed.";
      });
    }
  }
  function matches(d) {
    if (!filterText) return true;
    const q = filterText.toLowerCase();
    return (d.title + " " + d.artist + " " + d.tags).toLowerCase().indexOf(q) !== -1;
  }

  function buildConstellationEdges() {
    const n = points.length;
    if (n < 2 || n > 2200) return [];
    const K = n > 900 ? 1 : 2;
    const edgeSet = new Set();
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = points[i].x - points[j].x;
        const dy = points[i].y - points[j].y;
        row.push({ j: j, d2: dx * dx + dy * dy });
      }
      row.sort(function(a, b) { return a.d2 - b.d2; });
      for (let k = 0; k < Math.min(K, row.length); k++) {
        const j = row[k].j;
        const lo = i < j ? i : j;
        const hi = i < j ? j : i;
        edgeSet.add(lo + "," + hi);
      }
    }
    return Array.from(edgeSet).map(function(s) {
      const p = s.split(",");
      return [+p[0], +p[1]];
    });
  }

  function buildSectorCentroids() {
    const hints = DATA.clusterHints || [];
    const byC = d3.group(points, function(d) { return d.c; });
    const out = [];
    byC.forEach(function(pts, cid) {
      const mx = d3.mean(pts, function(d) { return d.x; });
      const my = d3.mean(pts, function(d) { return d.y; });
      var h = hints.find(function(x) { return x.id == cid; });
      var label = "sector " + cid;
      if (h && h.terms && h.terms.length) {
        label = h.terms.slice(0, 3).join(" · ");
      }
      if (label.length > 44) label = label.slice(0, 42) + "…";
      out.push({ x: mx, y: my, label: label, cid: cid });
    });
    return out;
  }

  const edgeList = buildConstellationEdges();
  const sectorNodes = buildSectorCentroids();

  const chart = d3.select("#chart");
  function chartSize() {
    const n = chart.node();
    return { w: Math.max(320, n.clientWidth || 800), h: Math.max(280, n.clientHeight || 560) };
  }
  let sz = chartSize();
  const svg = chart.append("svg").attr("width", sz.w).attr("height", sz.h);
  const g = svg.append("g");
  const gLinks = g.append("g").attr("class", "link-layer");
  const gSectors = g.append("g").attr("class", "sector-layer");
  const gTrip = g.append("g").attr("class", "trip-layer").style("display", "none");
  gTrip.append("path").attr("class", "trip-glow").attr("fill", "none");
  gTrip.append("path").attr("class", "trip-core").attr("fill", "none");
  gTrip.append("circle").attr("class", "trip-token").attr("r", 7).style("display", "none");
  const gDots = g.append("g");

  const xExt = d3.extent(points, function(d) { return d.x; });
  const yExt = d3.extent(points, function(d) { return d.y; });
  function pad(lo, hi) { return (hi - lo) * 0.08 || 1; }
  const xDom = [xExt[0] - pad(xExt[0], xExt[1]), xExt[1] + pad(xExt[0], xExt[1])];
  const yDom = [yExt[0] - pad(yExt[0], yExt[1]), yExt[1] + pad(yExt[0], yExt[1])];
  const margin = { l: 44, r: 16, t: 20, b: 28 };
  let x = d3.scaleLinear().domain(xDom).range([margin.l, sz.w - margin.r]);
  let y = d3.scaleLinear().domain(yDom).range([sz.h - margin.b, margin.t]);

  var tripOrder = null;
  var tripActive = false;
  var tripIdxSet = new Set();
  var TRIP_STOPS = 12;
  var flightActive = false;
  var flightIdx = -1;
  var flightFails = 0;
  var FLIGHT_MAX_FAILS = 3;

  function topKBySim(k) {
    var idx = points.map(function(_, i) { return i; });
    idx.sort(function(a, b) { return (points[b].sim || 0) - (points[a].sim || 0); });
    return idx.slice(0, k);
  }
  function greedySpatialOrder(indices) {
    if (indices.length <= 1) return indices.slice();
    var remaining = indices.slice();
    var startAt = 0;
    var bestSim = -Infinity;
    for (var t = 0; t < remaining.length; t++) {
      var s = points[remaining[t]].sim || 0;
      if (s > bestSim) { bestSim = s; startAt = t; }
    }
    var order = [remaining[startAt]];
    remaining.splice(startAt, 1);
    while (remaining.length) {
      var last = order[order.length - 1];
      var lx = points[last].x, ly = points[last].y;
      var bi = 0, bd = Infinity;
      for (var r = 0; r < remaining.length; r++) {
        var j = remaining[r];
        var dx = points[j].x - lx, dy = points[j].y - ly;
        var d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; bi = r; }
      }
      order.push(remaining[bi]);
      remaining.splice(bi, 1);
    }
    return order;
  }
  function computeGalacticTrip() {
    if (!useSim || !points.length || points[0].sim === undefined || points[0].sim === null) return null;
    return greedySpatialOrder(topKBySim(TRIP_STOPS));
  }
  function placeTripPath() {
    if (!tripOrder || tripOrder.length < 2) {
      gTrip.select("path.trip-glow").attr("d", null);
      gTrip.select("path.trip-core").attr("d", null);
      return;
    }
    var pts = tripOrder.map(function(i) { return [x(points[i].x), y(points[i].y)]; });
    var line = tripOrder.length < 3 ? d3.line() : d3.line().curve(d3.curveCatmullRom.alpha(0.75));
    var d0 = line(pts);
    gTrip.select("path.trip-glow").attr("d", d0);
    gTrip.select("path.trip-core").attr("d", d0);
  }
  function setTripFromIndices(order) {
    tripOrder = order;
    tripActive = !!(order && order.length);
    tripIdxSet = new Set(order || []);
    gTrip.style("display", tripActive ? null : "none");
    placeTripPath();
    gTrip.select(".trip-token").style("display", "none").interrupt();
    placeCircles();
  }
  function clearGalacticTrip() {
    if (flightActive) stopGalacticFlight({ silent: true });
    tripOrder = null;
    tripActive = false;
    tripIdxSet = new Set();
    gTrip.style("display", "none");
    gTrip.select("path.trip-glow").attr("d", null);
    gTrip.select("path.trip-core").attr("d", null);
    gTrip.select(".trip-token").style("display", "none").classed("flying", false).interrupt();
    placeCircles();
    var tw = document.getElementById("tripper-wrap");
    if (tw) { tw.style.display = "none"; tw.innerHTML = ""; }
  }
  function updateTripperPanel() {
    var wrap = document.getElementById("tripper-wrap");
    if (!wrap || !tripOrder || !tripOrder.length) return;
    wrap.style.display = "block";
    var flightBtn = flightActive
      ? "<button type=\\"button\\" id=\\"trip-flight\\" style=\\"padding:6px 10px;border-radius:6px;border:1px solid #da3633;background:#21262d;color:#ff7b72;cursor:pointer;font-size:12px;\\">Stop flight</button>"
      : "<button type=\\"button\\" id=\\"trip-flight\\" style=\\"padding:6px 10px;border-radius:6px;border:1px solid #a371f7;background:#21262d;color:#e6edf3;cursor:pointer;font-size:12px;\\">Take flight</button>";
    var subtitle = flightActive
      ? "In flight · stop " + (flightIdx + 1) + "/" + tripOrder.length + " · auto‑advances on track end"
      : tripOrder.length + " stops · top matches, path greedy in 2D";
    var html = "<h2 style=\\"font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin:0 0 8px;\\">Galactic tripper</h2>" +
      "<div style=\\"color:#8b949e;font-size:11px;margin-bottom:8px;\\">" + esc(subtitle) + "</div>" +
      "<div style=\\"display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;\\">" +
      flightBtn +
      "<button type=\\"button\\" id=\\"trip-preview\\" style=\\"padding:6px 10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#8b949e;cursor:pointer;font-size:12px;\\">Preview</button>";
    if (flightActive) {
      html += "<button type=\\"button\\" id=\\"trip-skip\\" style=\\"padding:6px 10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#8b949e;cursor:pointer;font-size:12px;\\">Skip stop</button>";
    }
    html += "<button type=\\"button\\" id=\\"trip-clear\\" style=\\"padding:6px 10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#8b949e;cursor:pointer;font-size:12px;\\">Clear</button></div>" +
      "<ol style=\\"margin:0;padding-left:18px;line-height:1.45;font-size:12px;max-height:220px;overflow-y:auto;\\">";
    for (var s = 0; s < tripOrder.length; s++) {
      var pi = tripOrder[s];
      var p = points[pi];
      var isNow = flightActive && s === flightIdx;
      var liBg = isNow ? "background:rgba(163,113,247,0.15);border-radius:4px;padding:2px 4px;" : "";
      var numColor = isNow ? "#f0f6fc" : "#a371f7";
      var titleColor = isNow ? "#f0f6fc" : "inherit";
      html += "<li style=\\"margin-bottom:4px;" + liBg + "\\">" +
        "<span style=\\"color:" + numColor + ";font-weight:600;\\">" + (s + 1) + ".</span> " +
        "<span style=\\"color:" + titleColor + ";\\">" + esc(p.title || "—") + "</span> " +
        "<span style=\\"color:#8b949e;\\">" + esc(p.artist || "") + "</span></li>";
    }
    html += "</ol>";
    wrap.innerHTML = html;
    document.getElementById("trip-flight").addEventListener("click", function() {
      if (flightActive) stopGalacticFlight();
      else startGalacticFlight();
    });
    document.getElementById("trip-preview").addEventListener("click", runTripPreview);
    var skipBtn = document.getElementById("trip-skip");
    if (skipBtn) skipBtn.addEventListener("click", function() {
      if (flightActive) playFlightStop(flightIdx + 1);
    });
    document.getElementById("trip-clear").addEventListener("click", clearGalacticTrip);
  }
  function startGalacticFlight() {
    if (!DATA.streamProxy) return;
    if (!tripOrder || tripOrder.length === 0) return;
    if (flightActive) return;
    flightActive = true;
    flightIdx = -1;
    flightFails = 0;
    zoomToTripBounds();
    var token = gTrip.select(".trip-token");
    token.interrupt();
    token.style("display", null).style("opacity", 1).classed("flying", true);
    playFlightStop(0);
    updateTripperPanel();
  }
  function stopGalacticFlight(opts) {
    if (!flightActive) return;
    opts = opts || {};
    flightActive = false;
    flightIdx = -1;
    flightFails = 0;
    try { streamAudio.pause(); } catch (e) {}
    gTrip.select(".trip-token").classed("flying", false);
    placeCircles();
    if (!opts.silent && audioDockSt) audioDockSt.textContent = "Flight stopped.";
    updateTripperPanel();
  }
  function finishFlight() {
    flightActive = false;
    flightIdx = -1;
    gTrip.select(".trip-token").classed("flying", false);
    placeCircles();
    if (audioDockSt) audioDockSt.textContent = "Flight complete · journey finished.";
    updateTripperPanel();
  }
  function playFlightStop(s) {
    if (!flightActive) return;
    if (!tripOrder) return;
    if (s >= tripOrder.length) { finishFlight(); return; }
    flightIdx = s;
    flightFails = 0;
    var d = points[tripOrder[s]];
    if (!d) { finishFlight(); return; }
    if (audioDockTitle) {
      audioDockTitle.textContent =
        "Stop " + (s + 1) + "/" + tripOrder.length + " · " +
        (d.title || "(no title)") + (d.artist ? (" — " + d.artist) : "");
    }
    var token = gTrip.select(".trip-token");
    token.interrupt();
    token.attr("transform", "translate(" + x(d.x) + "," + y(d.y) + ")");
    placeCircles();
    updateTripperPanel();
    if (!d.audioKey) { scheduleFlightSkip("No audio_key for this stop — skipping…"); return; }
    var hint = bvUnsupportedFormatHint(d.audioKey);
    if (hint) { scheduleFlightSkip("Unsupported format at this stop — skipping…"); return; }
    var mediaBase = (DATA.streamMediaPath || "/api/stream-audio").replace(/\\/$/, "");
    if (audioDockSt) audioDockSt.textContent = "In flight · loading…";
    streamAudio.src = mediaBase + "?assetName=" + encodeURIComponent(d.audioKey);
    var p = streamAudio.play();
    if (p !== undefined) {
      p.then(function() {
        if (flightActive && audioDockSt) audioDockSt.textContent = "In flight · playing stop " + (s + 1) + "/" + tripOrder.length + ".";
      }).catch(function(err) {
        if (!err) return;
        if (err.name === "AbortError") return;
        if (err.name === "NotAllowedError") {
          if (audioDockSt) audioDockSt.textContent = "Tap ▶ in the audio bar to continue the flight.";
          return;
        }
        if (audioDockSt) audioDockSt.textContent = err.message || "Playback failed.";
      });
    }
  }
  function scheduleFlightSkip(msg) {
    if (!flightActive) return;
    if (audioDockSt) audioDockSt.textContent = msg;
    setTimeout(function() {
      if (!flightActive) return;
      playFlightStop(flightIdx + 1);
    }, 1000);
  }
  function flightTokenTick() {
    if (!flightActive) return;
    if (!tripOrder || flightIdx < 0 || flightIdx >= tripOrder.length) return;
    var d = points[tripOrder[flightIdx]];
    var ni = flightIdx + 1;
    var nd = (ni < tripOrder.length) ? points[tripOrder[ni]] : null;
    var dur = streamAudio.duration;
    if (!nd || !isFinite(dur) || dur <= 0) return;
    var t = Math.max(0, Math.min(1, streamAudio.currentTime / dur));
    var tx = x(d.x) + (x(nd.x) - x(d.x)) * t;
    var ty = y(d.y) + (y(nd.y) - y(d.y)) * t;
    gTrip.select(".trip-token").attr("transform", "translate(" + tx + "," + ty + ")");
  }
  function zoomToTripBounds() {
    if (!tripOrder || !tripOrder.length) return;
    var xs = tripOrder.map(function(i) { return points[i].x; });
    var ys = tripOrder.map(function(i) { return points[i].y; });
    var minx = d3.min(xs), maxx = d3.max(xs), miny = d3.min(ys), maxy = d3.max(ys);
    var mx = (minx + maxx) / 2, my = (miny + maxy) / 2;
    var cw = Math.max(maxx - minx, 0.01), ch = Math.max(maxy - miny, 0.01);
    var span = Math.max(cw, ch);
    var pad = span * 0.45;
    var sc = Math.min(sz.w, sz.h) / (span + pad * 2) * 0.82;
    sc = Math.max(0.35, Math.min(sc, 10));
    var cx = x(mx), cy = y(my);
    var tf = d3.zoomIdentity.translate(sz.w / 2, sz.h / 2).scale(sc).translate(-cx, -cy);
    svg.transition().duration(1200).ease(d3.easeCubicInOut).call(zoomBeh.transform, tf);
  }
  function runTripPreview() {
    if (!tripOrder || tripOrder.length < 2) return;
    zoomToTripBounds();
    var pathNode = gTrip.select("path.trip-core").node();
    if (!pathNode || !pathNode.getTotalLength) return;
    var token = gTrip.select(".trip-token");
    token.interrupt();
    token.style("display", null).style("opacity", 1);
    var len = pathNode.getTotalLength();
    var dur = Math.min(14000, 2500 + tripOrder.length * 500);
    setTimeout(function() {
      token.transition().duration(0).attr("transform", function() {
        var p0 = pathNode.getPointAtLength(0);
        return "translate(" + p0.x + "," + p0.y + ")";
      });
      token.transition().duration(dur).ease(d3.easeLinear).attrTween("transform", function() {
        return function(t) {
          var p = pathNode.getPointAtLength(t * len);
          return "translate(" + p.x + "," + p.y + ")";
        };
      }).on("end", function() {
        token.transition().delay(600).style("opacity", 0).on("end", function() {
          token.style("display", "none").style("opacity", 1);
        });
      });
    }, 400);
  }

  function placeLinks() {
    gLinks.selectAll("line")
      .attr("x1", function(d) { return x(points[d[0]].x); })
      .attr("y1", function(d) { return y(points[d[0]].y); })
      .attr("x2", function(d) { return x(points[d[1]].x); })
      .attr("y2", function(d) { return y(points[d[1]].y); });
  }
  gLinks.selectAll("line").data(edgeList).join("line");
  placeLinks();

  function placeSectors() {
    gSectors.selectAll("text")
      .attr("x", function(d) { return x(d.x); })
      .attr("y", function(d) { return y(d.y); });
  }
  gSectors.selectAll("text").data(sectorNodes).join("text")
    .attr("text-anchor", "middle")
    .attr("dy", "0.35em")
    .text(function(d) { return d.label; });
  placeSectors();

  function placeCircles() {
    gDots.selectAll("circle")
      .attr("cx", function(d) { return x(d.x); })
      .attr("cy", function(d) { return y(d.y); })
      .attr("fill", dotFill)
      .attr("r", dotRadius)
      .attr("stroke-width", dotStrokeW)
      .attr("stroke", "rgba(255,255,255,0.28)")
      .attr("class", function(d, i) {
        var cls = "dot" + (matches(d) ? "" : " dim");
        if (tripActive) {
          if (tripIdxSet.has(i)) cls += " trip-stop";
          else cls += " trip-dim";
        }
        if (flightActive && flightIdx >= 0 && tripOrder && tripOrder[flightIdx] === i) {
          cls += " flight-now";
        }
        return cls;
      });
  }

  gDots.selectAll("circle").data(points).join("circle")
    .attr("class", "dot")
    .attr("r", dotRadius)
    .attr("stroke-width", dotStrokeW)
    .attr("stroke", "rgba(255,255,255,0.28)")
    .attr("fill", dotFill)
    .attr("cx", function(d) { return x(d.x); })
    .attr("cy", function(d) { return y(d.y); })
    .on("mouseenter", function(event, d) {
      if (!matches(d)) return;
      d3.select(this).attr("stroke-width", 2.2).attr("stroke", "rgba(240,246,252,0.75)");
      var simLine = (useSim && d.sim !== undefined) ? "<div style=\\"margin-top:6px;color:#58a6ff\\">cosine sim: " + d.sim.toFixed(4) + "</div>" : "";
      var streamLine = "";
      if (DATA.streamProxy && d.audioKey) {
        streamLine = "<div style=\\"margin-top:6px;color:#8b949e\\">Double-click to stream</div>";
        if (bvUnsupportedFormatHint(d.audioKey)) {
          streamLine += "<div style=\\"margin-top:4px;color:#d29922;font-size:11px\\">This file won’t play in Safari (FLAC/AIFF). Use Chrome/Edge/Firefox or an MP3.</div>";
        }
      }
      tooltip.style("display", "block")
        .html("<b>" + esc(d.title || "(no title)") + "</b><br>" + esc(d.artist || "") +
          "<div class=\\"tags\\">" + esc(d.tags || "") + "</div>" + simLine + streamLine +
          "<div style=\\"margin-top:6px;opacity:.7\\">" + esc(d.id || "") + "</div>")
        .style("left", (event.clientX + 14) + "px").style("top", (event.clientY + 14) + "px");
    })
    .on("mousemove", function(event) {
      tooltip.style("left", (event.clientX + 14) + "px").style("top", (event.clientY + 14) + "px");
    })
    .on("mouseleave", function(event, d) {
      tooltip.style("display", "none");
      d3.select(this).attr("stroke-width", dotStrokeW(d)).attr("stroke", "rgba(255,255,255,0.28)");
    })
    .on("dblclick", function(event, d) {
      event.preventDefault();
      event.stopPropagation();
      if (!matches(d)) return;
      playTrackFromDot(d);
    });

  const zoomBeh = d3.zoom()
    .filter(function(event) {
      if (event.type === "dblclick") return false;
      return (!event.ctrlKey && !event.button) || event.type === "wheel";
    })
    .scaleExtent([0.12, 48]).on("zoom", function(ev) {
    g.attr("transform", ev.transform);
  });
  svg.call(zoomBeh);
  svg.on("dblclick.zoom", null);

  function medianSim(arr) {
    var s = arr.slice().sort(function(a, b) { return a - b; });
    var m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function updateSignalReadout() {
    if (!useSim || !points.length || points[0].sim === undefined) {
      signalReadout.style("display", "none");
      return;
    }
    var sims = points.map(function(p) { return p.sim; });
    var lo = d3.min(sims);
    var hi = d3.max(sims);
    var med = medianSim(sims);
    var bins = 14;
    var hist = [];
    for (var bi = 0; bi < bins; bi++) hist.push(0);
    for (var i = 0; i < sims.length; i++) {
      var t = (sims[i] - lo) / (hi - lo + 1e-12);
      var b = Math.min(bins - 1, Math.floor(t * bins));
      hist[b]++;
    }
    var maxc = d3.max(hist) || 1;
    var w = 240, hbar = 40, pad = 2;
    var bw = (w - pad * 2) / bins;
    var bars = hist.map(function(c, i) {
      var bh = (c / maxc) * (hbar - 8);
      return "<rect x=\\"" + (pad + i * bw) + "\\" y=\\"" + (hbar - bh) + "\\" width=\\"" + (bw - 1) + "\\" height=\\"" + bh + "\\" fill=\\"rgba(88,166,255,0.45)\\"/>";
    }).join("");
    signalReadout.style("display", "block").html(
      "<h2 style=\\"margin:0 0 6px;font-size:11px;\\">Signal calibration</h2>" +
      "<div style=\\"color:#8b949e;font-size:11px;line-height:1.5\\">min · median · max<br/>" +
      lo.toFixed(4) + " · " + med.toFixed(4) + " · " + hi.toFixed(4) + "</div>" +
      "<svg width=\\"" + w + "\\" height=\\"" + hbar + "\\" style=\\"margin-top:6px\\">" + bars + "</svg>"
    );
  }

  function zoomToSignal() {
    if (!useSim || points[0].sim === undefined) return;
    var sorted = points.map(function(p) { return p.sim; }).sort(function(a, b) { return b - a; });
    var idx = Math.max(0, Math.floor(sorted.length * 0.12));
    var cutoff = sorted[idx];
    var sx = 0, sy = 0, sw = 0;
    for (var i = 0; i < points.length; i++) {
      if (points[i].sim >= cutoff) {
        sx += points[i].x * points[i].sim;
        sy += points[i].y * points[i].sim;
        sw += points[i].sim;
      }
    }
    if (sw < 1e-9) return;
    var mx = sx / sw, my = sy / sw;
    var cx = x(mx), cy = y(my);
    var sc = 2.35;
    var tf = d3.zoomIdentity.translate(sz.w / 2, sz.h / 2).scale(sc).translate(-cx, -cy);
    svg.transition().duration(880).ease(d3.easeCubicInOut).call(zoomBeh.transform, tf);
  }

  d3.select("#toggle-links").on("change", function() {
    gLinks.style("display", this.checked ? null : "none");
  });
  d3.select("#toggle-sectors").on("change", function() {
    gSectors.style("display", this.checked ? null : "none");
  });
  gLinks.style("display", d3.select("#toggle-links").property("checked") ? null : "none");
  gSectors.style("display", d3.select("#toggle-sectors").property("checked") ? null : "none");

  function onResize() {
    sz = chartSize();
    svg.attr("width", sz.w).attr("height", sz.h);
    x.range([margin.l, sz.w - margin.r]);
    y.range([sz.h - margin.b, margin.t]);
    placeLinks();
    placeSectors();
    placeTripPath();
    placeCircles();
  }
  window.addEventListener("resize", onResize);

  d3.select("#filter").on("input", function() {
    filterText = this.value.trim();
    placeCircles();
  });

  if (useSim) updateSignalReadout();

  if (DATA.dynamicSearch) {
    const LOG_KEY = "bon-voyage-mission-log";
    function pushMission(q) {
      try {
        var arr = JSON.parse(localStorage.getItem(LOG_KEY) || "[]");
        if (!Array.isArray(arr)) arr = [];
        arr = arr.filter(function(x) { return x !== q; });
        arr.unshift(q);
        arr = arr.slice(0, 8);
        localStorage.setItem(LOG_KEY, JSON.stringify(arr));
        renderMission();
      } catch (e) {}
    }
    function renderMission() {
      var el = document.getElementById("mission-log");
      var wrap = document.getElementById("mission-log-wrap");
      if (!el || !wrap) return;
      try {
        var arr = JSON.parse(localStorage.getItem(LOG_KEY) || "[]");
        el.innerHTML = "";
        arr.forEach(function(q) {
          var b = document.createElement("button");
          b.type = "button";
          b.textContent = q;
          b.addEventListener("click", function() {
            document.getElementById("voyage-q").value = q;
            runVoyageSearch();
          });
          el.appendChild(b);
        });
        wrap.style.display = arr.length ? "block" : "none";
      } catch (e) {}
    }
    const hdr = document.querySelector("header");
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;flex:1 1 360px;";
    row.innerHTML = '<span style="font-size:11px;color:#8b949e;text-transform:uppercase;">Voyage</span>' +
      '<input id="voyage-q" type="search" placeholder="Semantic search…" style="flex:1;min-width:160px;padding:8px 10px;border-radius:6px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:14px;"/>' +
      '<button id="voyage-go" type="button" style="padding:8px 14px;border-radius:6px;border:1px solid #238636;background:#238636;color:#fff;cursor:pointer;font-size:14px;">Search</button>' +
      '<span id="voyage-st" style="font-size:11px;color:#8b949e;max-width:220px;"></span>';
    hdr.appendChild(row);
    function runVoyageSearch() {
      const q = document.getElementById("voyage-q").value.trim();
      const st = document.getElementById("voyage-st");
      if (!q) { st.textContent = ""; return; }
      st.textContent = "Embedding…";
      fetch(DATA.apiSearchPath || "/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q })
      })
        .then(function(r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function(data) {
          if (data.error) { st.textContent = data.error; return; }
          const sims = data.similarities;
          if (!sims || sims.length !== points.length) {
            st.textContent = "Bad response (length mismatch)";
            return;
          }
          for (let i = 0; i < points.length; i++) points[i].sim = sims[i];
          useSim = true;
          refreshSimColor();
          refreshRScale();
          document.getElementById("title").textContent = DATA.title + " · " + q;
          st.textContent = "Done.";
          playSignalPing();
          pushMission(q);
          updateSignalReadout();
          var t = gDots.selectAll("circle")
            .transition()
            .duration(620)
            .ease(d3.easeCubicOut)
            .attr("fill", dotFill)
            .attr("r", dotRadius)
            .attr("stroke-width", dotStrokeW);
          if (typeof t.end === "function") {
            t.end().then(function() {
              placeCircles();
              zoomToSignal();
              afterTripLayout();
            }).catch(function() {
              placeCircles();
              zoomToSignal();
              afterTripLayout();
            });
          } else {
            placeCircles();
            zoomToSignal();
            afterTripLayout();
          }
          function afterTripLayout() {
            var gt = computeGalacticTrip();
            if (gt && gt.length) {
              setTripFromIndices(gt);
              updateTripperPanel();
            } else {
              clearGalacticTrip();
            }
          }
          hintEl.textContent = "Brighter & larger = closer match · scroll zoom · drag pan · filter · hover · tripper after search";
          const sv = document.getElementById("sidebar-voyage");
          if (sv) {
            sv.style.display = "block";
            sv.innerHTML = "<h2 style=\\"font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin:0 0 8px;\\">Scan lock</h2>" +
              "<div class=\\"cluster-block\\"><div style=\\"font-size:13px;line-height:1.4\\">" + esc(q) + "</div>" +
              "<div style=\\"margin-top:6px;color:#8b949e\\">Color & size encode cosine match to this signal.</div></div>";
          }
        })
        .catch(function() {
          st.textContent = "Request failed — open this page from embed_pipeline.py cosmos (not file://)";
        });
    }
    document.getElementById("voyage-go").addEventListener("click", runVoyageSearch);
    document.getElementById("voyage-q").addEventListener("keydown", function(ev) {
      if (ev.key === "Enter") runVoyageSearch();
    });
    renderMission();
  }
})();
  </script>
</body>
</html>
"""
