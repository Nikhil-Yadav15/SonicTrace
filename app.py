# app.py

"""
SonicTrace - Streamlit Web Interface
Speaker diarization, overlap detection, emotion recognition and transcription
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import Config
from core.audio_processor import audio_processor
from core.transcriber import transcriber
from utils.audio_loader import audio_loader

st.set_page_config(
    page_title="SonicTrace",
    page_icon="🎙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 2.2rem; }
    h1 { letter-spacing: -0.02em; }
    .app-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0; }
    .app-tagline { color: #8B95A7; font-size: 0.95rem; margin-top: 0.15rem; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    [data-testid="stMetricLabel"] { color: #8B95A7; }
</style>
""",
    unsafe_allow_html=True,
)

# Consistent colors across all charts
SPEAKER_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756",
    "#72B7B2", "#EECA3B", "#FF9DA6", "#9D755D", "#BAB0AC",
]
OVERLAP_COLOR = "#8B95A7"
EMOTION_COLORS = {
    "angry": "#E45756",
    "calm": "#72B7B2",
    "disgust": "#54A24B",
    "fearful": "#B279A2",
    "happy": "#EECA3B",
    "neutral": "#8B95A7",
    "sad": "#4C78A8",
    "surprised": "#F58518",
    "fear": "#B279A2",
    "surprise": "#F58518",
    "unknown": "#5C6470",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=12),
)


def init_session_state():
    defaults = {
        "results": None,
        "audio_bytes": None,
        "audio_name": None,
        "waveform_preview": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def speaker_name(seg):
    """Human-readable speaker label for a segment."""
    if seg.get("is_overlap"):
        return "Overlap"
    if seg.get("speaker") is not None:
        return f"Speaker {seg['speaker']}"
    return "Unassigned"


def speaker_sort_key(name):
    if name.startswith("Speaker "):
        suffix = name.split(" ", 1)[1]
        return (0, int(suffix)) if suffix.isdigit() else (1, 0)
    return (2, 0) if name == "Overlap" else (1, 0)


def sidebar_controls():
    st.sidebar.markdown("### Analysis setup")

    uploaded_file = st.sidebar.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="WAV, MP3, M4A, FLAC or OGG",
    )

    st.sidebar.markdown("#### Pipeline stages")
    enable_diarization = st.sidebar.checkbox(
        "Speaker diarization", value=True,
        help="Group speech segments by speaker (who spoke when)",
    )
    enable_overlap = st.sidebar.checkbox(
        "Overlap detection", value=True,
        help="Flag regions where multiple people talk at once",
    )
    enable_emotion = st.sidebar.checkbox(
        "Emotion recognition", value=True,
        help="Classify the emotional tone of each segment",
    )
    enable_transcription = st.sidebar.checkbox(
        "Transcription", value=True,
        help="Convert speech to text with Whisper",
    )

    n_speakers = None
    if enable_diarization:
        auto_speakers = st.sidebar.checkbox(
            "Detect speaker count automatically", value=True
        )
        if not auto_speakers:
            n_speakers = st.sidebar.slider("Number of speakers", 2, 10, 2)

    return uploaded_file, {
        "enable_overlap": enable_overlap,
        "enable_diarization": enable_diarization,
        "enable_emotion": enable_emotion,
        "enable_transcription": enable_transcription,
        "n_speakers": n_speakers,
    }


def build_waveform_preview(waveform, sample_rate, points=2400):
    """Downsample a waveform to a min/max envelope for plotting."""
    n = len(waveform)
    if n == 0:
        return None
    bin_size = max(1, n // points)
    usable = (n // bin_size) * bin_size
    chunks = waveform[:usable].reshape(-1, bin_size)
    return {
        "times": (np.arange(chunks.shape[0]) * bin_size / sample_rate).tolist(),
        "upper": chunks.max(axis=1).tolist(),
        "lower": chunks.min(axis=1).tolist(),
        "duration": n / sample_rate,
    }


def process_audio_file(uploaded_file, options):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        waveform, sr = audio_loader.load(tmp_path)
        st.session_state.waveform_preview = build_waveform_preview(waveform, sr)

        results = audio_processor.process(
            tmp_path,
            enable_overlap=options["enable_overlap"],
            enable_diarization=options["enable_diarization"],
            enable_emotion=options["enable_emotion"],
            enable_transcription=options["enable_transcription"],
            n_speakers=options["n_speakers"],
            show_progress=True,
        )
        return results
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def display_summary(results):
    summary = results["summary"]
    metadata = results["metadata"]

    cols = st.columns(5)
    cols[0].metric("Duration", f"{metadata['duration']:.1f} s")
    cols[1].metric("Segments", summary["total_segments"])
    cols[2].metric("Speakers", summary.get("n_speakers", 0))
    cols[3].metric("Words", summary.get("total_words", 0))
    cols[4].metric("Processing time", f"{metadata['processing_time']:.1f} s")


def display_waveform(segments):
    preview = st.session_state.waveform_preview
    if not preview:
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preview["times"] + preview["times"][::-1],
            y=preview["upper"] + preview["lower"][::-1],
            fill="toself",
            mode="lines",
            line=dict(width=0.4, color="#4C78A8"),
            fillcolor="rgba(76, 120, 168, 0.45)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for seg in segments:
        fig.add_vrect(
            x0=seg["start"], x1=seg["end"],
            fillcolor="rgba(245, 133, 24, 0.12)" if seg.get("is_overlap")
            else "rgba(84, 162, 75, 0.10)",
            line_width=0,
        )
    fig.update_layout(
        height=170,
        margin=dict(t=10, l=0, r=0, b=0),
        xaxis=dict(title="Time (s)", range=[0, preview["duration"]]),
        yaxis=dict(visible=False),
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Waveform with detected speech regions (orange tint = overlap).")


def display_timeline(results):
    segments = results["segments"]
    if not segments:
        st.info("No segments to display.")
        return

    display_waveform(segments)

    rows = {}
    for seg in segments:
        rows.setdefault(speaker_name(seg), []).append(seg)

    row_names = sorted(rows, key=speaker_sort_key)
    palette = {}
    color_idx = 0
    for name in row_names:
        if name == "Overlap":
            palette[name] = OVERLAP_COLOR
        else:
            palette[name] = SPEAKER_PALETTE[color_idx % len(SPEAKER_PALETTE)]
            color_idx += 1

    fig = go.Figure()
    for name in row_names:
        segs = rows[name]
        hover = [
            f"{name}<br>{s['start']:.2f}s – {s['end']:.2f}s"
            f"<br>Emotion: {s.get('emotion', 'n/a')}"
            + (f"<br>“{s.get('text', '')[:80]}”" if s.get("text") else "")
            for s in segs
        ]
        fig.add_trace(
            go.Bar(
                y=[name] * len(segs),
                x=[s["duration"] for s in segs],
                base=[s["start"] for s in segs],
                orientation="h",
                marker=dict(
                    color=palette[name],
                    line=dict(width=0),
                    pattern=dict(shape="/" if name == "Overlap" else ""),
                ),
                width=0.55,
                name=name,
                hovertext=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=120 + 60 * len(row_names),
        xaxis=dict(title="Time (s)", range=[0, results["metadata"]["duration"]]),
        yaxis=dict(
            categoryorder="array",
            categoryarray=row_names[::-1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, l=0, r=0, b=0),
        **PLOT_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def display_speakers(results):
    speakers = results["statistics"].get("speakers", {}).get("speakers", {})
    if not speakers:
        st.info("Speaker diarization was not run or found no speakers.")
        return

    speaker_rows = []
    for label, stats in sorted(speakers.items(), key=lambda kv: speaker_sort_key(f"Speaker {kv[0]}")):
        speaker_rows.append(
            {
                "Speaker": f"Speaker {label}",
                "Segments": stats["segments"],
                "Speaking time (s)": round(stats["total_duration"], 2),
                "Share of speech": f"{stats['percentage']:.1f} %",
            }
        )
    df = pd.DataFrame(speaker_rows)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        fig = go.Figure(
            go.Pie(
                labels=df["Speaker"],
                values=[
                    speakers[row["Speaker"].split(" ", 1)[1]]["total_duration"]
                    for row in speaker_rows
                ],
                hole=0.55,
                marker=dict(colors=SPEAKER_PALETTE[: len(df)]),
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            height=280,
            showlegend=False,
            margin=dict(t=10, l=10, r=10, b=10),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-speaker emotion mix (bonus task)
    segments = [s for s in results["segments"] if s.get("speaker") and s.get("emotion")]
    if segments:
        st.markdown("##### Emotion mix per speaker")
        mix = {}
        for seg in segments:
            key = f"Speaker {seg['speaker']}"
            mix.setdefault(key, {})
            mix[key][seg["emotion"]] = mix[key].get(seg["emotion"], 0) + seg["duration"]

        speakers_sorted = sorted(mix, key=speaker_sort_key)
        emotions = sorted({e for m in mix.values() for e in m})
        fig = go.Figure()
        for emotion in emotions:
            fig.add_trace(
                go.Bar(
                    x=speakers_sorted,
                    y=[mix[s].get(emotion, 0) for s in speakers_sorted],
                    name=emotion,
                    marker_color=EMOTION_COLORS.get(emotion, "#5C6470"),
                )
            )
        fig.update_layout(
            barmode="stack",
            height=300,
            yaxis_title="Speaking time (s)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, l=0, r=0, b=0),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_emotions(results):
    emotions = results["statistics"].get("emotions", {}).get("emotions", {})
    if not emotions:
        st.info("Emotion recognition was not run.")
        return

    ordered = sorted(emotions.items(), key=lambda kv: kv[1]["percentage"], reverse=True)
    emotion_rows = [
        {
            "Emotion": emotion.capitalize(),
            "Segments": stats["count"],
            "Duration (s)": round(stats["duration"], 2),
            "Share": f"{stats['percentage']:.1f} %",
            "Avg. confidence": f"{stats['avg_confidence'] * 100:.0f} %",
        }
        for emotion, stats in ordered
    ]
    df = pd.DataFrame(emotion_rows)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        fig = go.Figure(
            go.Bar(
                x=[emotion.capitalize() for emotion, _ in ordered],
                y=[stats["percentage"] for _, stats in ordered],
                marker_color=[
                    EMOTION_COLORS.get(emotion.lower(), "#5C6470")
                    for emotion, _ in ordered
                ],
            )
        )
        fig.update_layout(
            height=300,
            yaxis_title="Share of segments (%)",
            margin=dict(t=10, l=0, r=0, b=0),
            **PLOT_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)


def display_segments(results):
    segments = results["segments"]
    if not segments:
        st.info("No segments to display.")
        return

    speaker_options = sorted(
        {speaker_name(seg) for seg in segments}, key=speaker_sort_key
    )
    emotion_options = sorted(
        {seg.get("emotion", "unknown") for seg in segments if seg.get("emotion")}
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        filter_speaker = st.selectbox("Speaker", ["All"] + speaker_options)
    with col2:
        filter_emotion = st.selectbox("Emotion", ["All"] + emotion_options)
    with col3:
        show_overlap_only = st.checkbox("Overlaps only", value=False)

    filtered = segments
    if filter_speaker != "All":
        filtered = [s for s in filtered if speaker_name(s) == filter_speaker]
    if filter_emotion != "All":
        filtered = [s for s in filtered if s.get("emotion") == filter_emotion]
    if show_overlap_only:
        filtered = [s for s in filtered if s.get("is_overlap")]

    st.caption(f"Showing {min(len(filtered), 100)} of {len(filtered)} matching segments "
               f"({len(segments)} total).")

    for seg in filtered[:100]:
        with st.container(border=True):
            col_time, col_body = st.columns([1, 4.2])
            with col_time:
                st.markdown(f"**{seg['start']:.2f} – {seg['end']:.2f} s**")
                st.caption(f"{seg['duration']:.2f} s")
            with col_body:
                meta = [speaker_name(seg)]
                if seg.get("emotion"):
                    confidence = seg.get("emotion_confidence", 0)
                    meta.append(f"{seg['emotion']} ({confidence * 100:.0f} %)")
                if seg.get("is_overlap"):
                    meta.append("overlapping speech")
                st.markdown(" · ".join(meta))

                text = (seg.get("text") or "").strip()
                if text:
                    st.markdown(f"> {text}")
                else:
                    st.caption("No transcription for this segment.")


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def display_downloads(results):
    base_name = Path(results["metadata"]["filename"]).stem or "sonictrace"

    json_data = json.dumps(
        convert_to_serializable(results), indent=2, ensure_ascii=False
    )
    txt_data = transcriber.get_full_transcription(
        results["segments"],
        include_timestamps=True,
        include_speaker=True,
        include_emotion=True,
    )
    srt_data = transcriber.export_transcription(results["segments"], format="srt")
    vtt_data = transcriber.export_transcription(results["segments"], format="vtt")

    col1, col2, col3, col4 = st.columns(4)
    col1.download_button(
        "JSON — full results", json_data,
        file_name=f"{base_name}_results.json", mime="application/json",
        use_container_width=True,
    )
    col2.download_button(
        "TXT — transcript", txt_data,
        file_name=f"{base_name}_transcript.txt", mime="text/plain",
        use_container_width=True,
    )
    col3.download_button(
        "SRT — subtitles", srt_data,
        file_name=f"{base_name}.srt", mime="text/plain",
        use_container_width=True,
    )
    col4.download_button(
        "VTT — subtitles", vtt_data,
        file_name=f"{base_name}.vtt", mime="text/vtt",
        use_container_width=True,
    )


def display_landing():
    st.info("Upload an audio file in the sidebar, choose the pipeline stages, "
            "and press **Analyze** to get started.")

    st.markdown("#### How it works")
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("1 · Voice activity detection",
         "Silero VAD finds the regions of the recording that contain speech."),
        ("2 · Speaker embeddings",
         "Each speech segment is converted into a vector that captures the "
         "voice's characteristics."),
        ("3 · Clustering",
         "Segments with similar voices are grouped — no enrollment or voice "
         "profiles needed. Each cluster becomes Speaker 1, Speaker 2, …"),
        ("4 · Emotion & transcript",
         "Every segment is tagged with an emotion and transcribed with "
         "Whisper, then everything is laid out on a timeline."),
    ]
    for col, (title, body) in zip((col1, col2, col3, col4), steps):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(body)


def main():
    init_session_state()

    st.markdown('<p class="app-title">SonicTrace</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-tagline">Who spoke when, and how they felt — '
        "speaker diarization, overlap detection, emotion recognition and "
        "transcription for any recording.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    uploaded_file, options = sidebar_controls()

    if uploaded_file is not None:
        st.sidebar.divider()
        if st.sidebar.button("Analyze", type="primary", use_container_width=True):
            with st.spinner(
                "Processing audio — the first run downloads models and can "
                "take a few minutes…"
            ):
                try:
                    results = process_audio_file(uploaded_file, options)
                    st.session_state.results = results
                    st.session_state.audio_bytes = uploaded_file.getvalue()
                    st.session_state.audio_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.exception(e)

    results = st.session_state.results

    if results is None:
        display_landing()
        return

    if st.session_state.audio_bytes:
        st.caption(st.session_state.audio_name or results["metadata"]["filename"])
        st.audio(st.session_state.audio_bytes)

    display_summary(results)
    st.divider()

    if results["summary"]["total_segments"] == 0:
        st.warning("No speech was detected in this recording.")
        return

    tab_timeline, tab_speakers, tab_emotions, tab_segments, tab_export = st.tabs(
        ["Timeline", "Speakers", "Emotions", "Segments", "Export"]
    )

    with tab_timeline:
        display_timeline(results)
    with tab_speakers:
        display_speakers(results)
    with tab_emotions:
        display_emotions(results)
    with tab_segments:
        display_segments(results)
    with tab_export:
        display_downloads(results)


if __name__ == "__main__":
    main()
