# app.py

"""
SonicTrace - Streamlit Web Interface
Complete audio analysis dashboard
"""

import streamlit as st
import tempfile
from pathlib import Path
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import Config
from core.audio_processor import audio_processor
from utils.audio_loader import audio_loader
import numpy as np

# Page config
st.set_page_config(
    page_title="SonicTrace - Audio Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    :root {
        --primary: #6366F1;
        --primary-soft: rgba(99, 102, 241, 0.12);
        --primary-strong: #4F46E5;
        --accent: #EC4899;
        --bg-soft: #0F172A;
        --bg-elevated: #020617;
        --border-subtle: rgba(148, 163, 184, 0.4);
        --text-main: #E5E7EB;
        --text-muted: #9CA3AF;
        --card-bg: rgba(15, 23, 42, 0.9);
    }

    /* Global background */
    .stApp {
        background: radial-gradient(circle at top left, #1e293b 0, #020617 45%, #020617 100%);
        color: var(--text-main);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }

    /* Main header */
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(120deg, #A855F7 0%, #6366F1 40%, #22C55E 80%, #F97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        text-align: center;
        color: var(--text-muted);
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    /* Divider */
    .main-divider {
        margin-top: 0.75rem;
        margin-bottom: 1.75rem;
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.4);
    }

    /* Section title */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    .section-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.95rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.14);
        color: var(--text-muted);
        margin-left: 0.4rem;
    }

    /* Metric cards */
    .metrics-row {
        margin-bottom: 0.75rem;
    }
    .metric-card {
        background: radial-gradient(circle at top left, rgba(148, 163, 184, 0.09) 0, rgba(15, 23, 42, 0.95) 45%, rgba(15, 23, 42, 1) 100%);
        border-radius: 1rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.75);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.15rem;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 600;
    }
    .metric-subtext {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.15rem;
    }

    /* Generic card container */
    .card {
        background: var(--card-bg);
        border-radius: 1.1rem;
        padding: 1.1rem 1.2rem;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.85);
        margin-bottom: 0.9rem;
    }

    .card-compact {
        background: var(--card-bg);
        border-radius: 0.9rem;
        padding: 0.85rem 0.9rem;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.85);
        margin-bottom: 0.5rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #020617 40%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .sidebar-subtitle {
        font-size: 0.95rem;
        font-weight: 500;
    }
    .sidebar-block {
        padding: 0.6rem 0.3rem 0.9rem 0.1rem;
        border-radius: 0.9rem;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(51, 65, 85, 0.9);
        margin-bottom: 0.7rem;
    }

    /* Upload area */
    .upload-label {
        font-size: 0.88rem;
        color: var(--text-muted);
        margin-bottom: 0.1rem;
    }
    .upload-helper {
        font-size: 0.78rem;
        color: var(--text-muted);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 0.4rem 0.9rem !important;
        font-size: 0.88rem !important;
        border: 1px solid transparent !important;
        margin-right: 0.25rem !important;
    }
    button[aria-selected="true"][data-baseweb="tab"] {
        background: linear-gradient(120deg, #4F46E5 0%, #6366F1 50%, #EC4899 100%) !important;
        color: white !important;
        border-color: rgba(129, 140, 248, 0.7) !important;
    }
    button[aria-selected="false"][data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.9) !important;
        color: var(--text-muted) !important;
        border-color: rgba(30, 64, 175, 0.45) !important;
    }

    /* Dataframe tweaks */
    .stDataFrame {
        border-radius: 0.75rem;
        overflow: hidden;
        border: 1px solid rgba(30, 64, 175, 0.55);
    }

    /* Segment chips */
    .segment-meta {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.12rem 0.55rem;
        font-size: 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(15, 23, 42, 0.9);
        color: var(--text-muted);
    }
    .badge-primary {
        border-color: rgba(129, 140, 248, 0.8);
        background: rgba(79, 70, 229, 0.12);
        color: #E5E7EB;
    }
    .badge-danger {
        border-color: rgba(248, 113, 113, 0.9);
        background: rgba(248, 113, 113, 0.18);
        color: #FCA5A5;
    }
    .badge-soft {
        border-style: dashed;
        opacity: 0.9;
    }

    .segment-time {
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    .segment-text {
        font-size: 0.94rem;
        line-height: 1.45;
        margin-top: 0.1rem;
    }

    /* Download buttons row */
    .download-row {
        margin-top: 0.5rem;
    }

    /* Feature bullets */
    .feature-card {
        background: radial-gradient(circle at top left, rgba(79, 70, 229, 0.15) 0, rgba(15, 23, 42, 0.98) 45%, rgba(15, 23, 42, 1) 100%);
        border-radius: 1.1rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(55, 65, 81, 0.95);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.9);
    }
    .feature-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }
    .feature-list {
        padding-left: 1.1rem;
        margin-bottom: 0;
    }
    .feature-list li {
        margin-bottom: 0.32rem;
        font-size: 0.9rem;
    }

    /* Info box refinement */
    .stAlert {
        border-radius: 0.9rem;
        border: 1px solid rgba(79, 70, 229, 0.5);
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables"""
    if "results" not in st.session_state:
        st.session_state.results = None
    if "processing" not in st.session_state:
        st.session_state.processing = False


def display_header():
    """Display app header"""
    st.markdown('<h1 class="main-header">SonicTrace</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="main-subtitle">'
        '🎤 Advanced Audio Intelligence — Voice Activity • Speaker Diarization • Emotion • Transcription'
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="main-divider" />', unsafe_allow_html=True)


def sidebar_settings():
    """Sidebar with upload and settings"""
    st.sidebar.markdown('<div class="sidebar-title">⚙️ Control Panel</div>', unsafe_allow_html=True)

    # File upload
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-subtitle">📁 Upload Audio</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="upload-label">Choose an audio file</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.sidebar.file_uploader(
        "",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Supported formats: WAV, MP3, M4A, FLAC, OGG",
    )

    st.sidebar.markdown("---")

    # Processing options
    st.sidebar.markdown(
        '<div class="sidebar-subtitle">🔧 Analysis Options</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar.container():
        enable_overlap = st.checkbox(
            "Overlap Detection",
            value=True,
            help="Detect when multiple speakers talk simultaneously",
        )

        enable_diarization = st.checkbox(
            "Speaker Diarization",
            value=True,
            help="Identify and separate different speakers",
        )

        enable_emotion = st.checkbox(
            "Emotion Recognition",
            value=True,
            help="Detect emotions in speech",
        )

        enable_transcription = st.checkbox(
            "Speech Transcription",
            value=True,
            help="Convert speech to text",
        )

    # Speaker settings
    if enable_diarization:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '<div class="sidebar-subtitle">👥 Speaker Settings</div>',
            unsafe_allow_html=True,
        )

        with st.sidebar.expander("Speaker configuration", expanded=True):
            auto_speakers = st.checkbox(
                "Auto-detect speakers",
                value=True,
                help="Automatically determine number of speakers",
            )

            if not auto_speakers:
                n_speakers = st.slider(
                    "Number of speakers",
                    min_value=2,
                    max_value=10,
                    value=2,
                )
            else:
                n_speakers = None
    else:
        n_speakers = None

    return uploaded_file, {
        "enable_overlap": enable_overlap,
        "enable_diarization": enable_diarization,
        "enable_emotion": enable_emotion,
        "enable_transcription": enable_transcription,
        "n_speakers": n_speakers,
    }


def process_audio_file(uploaded_file, options):
    """Process uploaded audio file"""

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Create a container for logs
        log_container = st.container()

        with log_container:
            st.info("🎧 Processing audio… Detailed logs are available in your terminal.")

        # Process
        results = audio_processor.process(
            tmp_path,
            enable_overlap=options["enable_overlap"],
            enable_diarization=options["enable_diarization"],
            enable_emotion=options["enable_emotion"],
            enable_transcription=options["enable_transcription"],
            n_speakers=options["n_speakers"],
            show_progress=True,  # Enable detailed logging
        )

        return results

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        st.exception(e)
        raise

    finally:
        # Cleanup temp file
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass


def display_summary(results):
    """Display summary statistics"""
    st.markdown(
        '<div class="section-title">📊 Session Summary'
        '<span class="section-pill">High-level view</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    summary = results["summary"]
    metadata = results["metadata"]

    st.markdown('<div class="metrics-row">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Duration</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{metadata["duration"]:.1f}s</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-subtext">Total audio length</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Segments</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{summary["total_segments"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-subtext">Voice activity chunks</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Speakers</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{summary.get("n_speakers", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-subtext">Detected unique speakers</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Words</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{summary.get("total_words", 0)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-subtext">Transcribed word count</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="metric-label">Processing Time</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metric-value">{metadata["processing_time"]:.1f}s</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-subtext">End-to-end pipeline time</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def display_speaker_breakdown(results):
    """Display speaker statistics"""

    if "speakers" not in results["statistics"]:
        return

    speakers = results["statistics"]["speakers"].get("speakers", {})

    if not speakers:
        return

    st.markdown(
        '<div class="section-title">👥 Speaker Breakdown'
        '<span class="section-pill">Who spoke and for how long</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Create dataframe
    speaker_data = []
    for speaker, stats in speakers.items():
        speaker_data.append(
            {
                "Speaker": f"Speaker {speaker}",
                "Segments": stats["segments"],
                "Duration (s)": round(stats["total_duration"], 2),
                "Percentage": round(stats["percentage"], 1),
            }
        )

    df = pd.DataFrame(speaker_data)

    col1, col2 = st.columns([2.3, 1.7])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.pie(
            df,
            values="Duration (s)",
            names="Speaker",
            title="Speaking Time Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            margin=dict(t=60, l=0, r=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def display_emotion_breakdown(results):
    """Display emotion statistics"""

    if "emotions" not in results["statistics"]:
        return

    emotions = results["statistics"]["emotions"].get("emotions", {})

    if not emotions:
        return

    st.markdown(
        '<div class="section-title">🎭 Emotion Distribution'
        '<span class="section-pill">Emotional tone across the session</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Create dataframe
    emotion_data = []
    for emotion, stats in emotions.items():
        emotion_data.append(
            {
                "Emotion": emotion.capitalize(),
                "Count": stats["count"],
                "Duration (s)": round(stats["duration"], 2),
                "Percentage": round(stats["percentage"], 1),
                "Confidence": round(stats["avg_confidence"] * 100, 1),
            }
        )

    df = pd.DataFrame(emotion_data).sort_values("Percentage", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.bar(
            df,
            x="Emotion",
            y="Percentage",
            title="Emotion Distribution (%)",
            color="Emotion",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=60, l=0, r=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def display_timeline(results):
    """Display interactive timeline"""

    segments = results["segments"]

    if not segments:
        return

    st.markdown(
        '<div class="section-title">⏱️ Interactive Timeline'
        '<span class="section-pill">Explore speech over time</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Create timeline data
    timeline_data = []

    for seg in segments:
        # Determine color based on speaker/emotion
        speaker = seg.get("speaker", "Unknown")
        emotion = seg.get("emotion", "neutral")
        text = seg.get("text", "")[:50]  # First 50 chars

        timeline_data.append(
            {
                "Speaker": f"Speaker {speaker}" if speaker != "Unknown" else "Unknown",
                "Start": seg["start"],
                "End": seg["end"],
                "Duration": seg["duration"],
                "Emotion": emotion.capitalize(),
                "Text": text,
                "Overlap": seg.get("is_overlap", False),
            }
        )

    df = pd.DataFrame(timeline_data)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Speaker",
        color="Emotion",
        hover_data=["Text", "Duration", "Overlap"],
        title="Speech Timeline",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Speaker",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, l=0, r=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def display_segments(results):
    """Display detailed segments"""

    segments = results["segments"]

    if not segments:
        st.info("No segments to display")
        return

    st.markdown(
        '<div class="section-title">📝 Detailed Segments'
        '<span class="section-pill">Fine-grained view</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Filter options
    filter_container = st.container()
    with filter_container:
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_speaker = st.selectbox(
                "Filter by Speaker",
                ["All"] + sorted(set(seg.get("speaker", "Unknown") for seg in segments)),
            )

        with col2:
            filter_emotion = st.selectbox(
                "Filter by Emotion",
                ["All"] + sorted(set(seg.get("emotion", "neutral") for seg in segments)),
            )

        with col3:
            show_overlap_only = st.checkbox("Show Overlaps Only", value=False)

    # Filter segments
    filtered_segments = segments

    if filter_speaker != "All":
        filtered_segments = [s for s in filtered_segments if s.get("speaker") == filter_speaker]

    if filter_emotion != "All":
        filtered_segments = [s for s in filtered_segments if s.get("emotion") == filter_emotion]

    if show_overlap_only:
        filtered_segments = [s for s in filtered_segments if s.get("is_overlap", False)]

    st.markdown(
        f"<span class='segment-time'>Showing <b>{len(filtered_segments)}</b> of {len(segments)} segments</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Display segments
    for idx, seg in enumerate(filtered_segments[:50]):  # Limit to 50
        st.markdown('<div class="card-compact">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(
                f"<div class='segment-time'>"
                f"[{seg['start']:.2f}s → {seg['end']:.2f}s]<br>"
                f"<span class='segment-time'>Duration: {seg['duration']:.2f}s</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col2:
            # Header line
            header_parts = []

            if "speaker" in seg:
                header_parts.append(
                    f"<span class='badge badge-primary'>🗣️ Speaker {seg['speaker']}</span>"
                )

            if "emotion" in seg:
                emotion_emoji = {
                    "happy": "😊",
                    "sad": "😢",
                    "angry": "😠",
                    "fear": "😨",
                    "neutral": "😐",
                    "surprise": "😲",
                    "disgust": "🤢",
                    "calm": "😌",
                    "fearful": "😨",
                    "surprised": "😲",
                }
                emoji = emotion_emoji.get(seg["emotion"], "😐")
                header_parts.append(
                    f"<span class='badge badge-soft'>{emoji} {seg['emotion'].capitalize()}</span>"
                )

            if seg.get("is_overlap"):
                header_parts.append("<span class='badge badge-danger'>🔀 Overlap</span>")

            st.markdown(
                f"<div class='segment-meta'>{''.join(header_parts)}</div>",
                unsafe_allow_html=True,
            )

            # Transcription
            text = seg.get("text", "").strip()
            if text:
                st.markdown(
                    f"<div class='segment-text'>💬 {text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("_No transcription_")

        st.markdown("</div>", unsafe_allow_html=True)


def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
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
    else:
        return obj


def display_download_options(results):
    """Display download options"""

    st.markdown(
        '<div class="section-title">💾 Export'
        '<span class="section-pill">Take your results anywhere</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card download-row">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # JSON
    with col1:
        # Convert numpy types to native Python types
        serializable_results = convert_to_serializable(results)
        json_data = json.dumps(serializable_results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 JSON (Full Results)",
            data=json_data,
            file_name="sonictrace_results.json",
            mime="application/json",
            use_container_width=True,
        )

    # TXT
    with col2:
        from core.transcriber import transcriber

        txt_data = transcriber.get_full_transcription(
            results["segments"],
            include_timestamps=True,
            include_speaker=True,
            include_emotion=True,
        )
        st.download_button(
            label="📝 TXT (Transcript)",
            data=txt_data,
            file_name="sonictrace_transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # SRT
    with col3:
        from core.transcriber import transcriber

        srt_data = transcriber.export_transcription(results["segments"], format="srt")
        st.download_button(
            label="🎬 SRT (Subtitles)",
            data=srt_data,
            file_name="sonictrace_subtitles.srt",
            mime="text/plain",
            use_container_width=True,
        )

    # VTT
    with col4:
        from core.transcriber import transcriber

        vtt_data = transcriber.export_transcription(results["segments"], format="vtt")
        st.download_button(
            label="📺 VTT (WebVTT)",
            data=vtt_data,
            file_name="sonictrace_subtitles.vtt",
            mime="text/vtt",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main app function"""

    init_session_state()
    display_header()

    # Sidebar
    uploaded_file, options = sidebar_settings()

    # Process button
    if uploaded_file is not None:
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Process Audio", type="primary"):
            with st.spinner("Processing audio..."):
                try:
                    results = process_audio_file(uploaded_file, options)
                    st.session_state.results = results
                    st.success("✅ Processing complete!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)

    # Display results
    if st.session_state.results is not None:
        results = st.session_state.results

        # Summary
        display_summary(results)

        st.markdown("---")

        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⏱️ Timeline", "📝 Segments", "💾 Download"])

        with tab1:
            display_speaker_breakdown(results)
            st.markdown("---")
            display_emotion_breakdown(results)

        with tab2:
            display_timeline(results)

        with tab3:
            display_segments(results)

        with tab4:
            display_download_options(results)

    else:
        # Landing page
        st.info("👈 Upload an audio file from the sidebar to begin a SonicTrace analysis.")

        st.markdown("### ✨ What SonicTrace can do")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
<div class="feature-card">
  <div class="feature-title">Speech & Speaker Intelligence</div>
  <ul class="feature-list">
    <li>🎤 <b>Voice Activity Detection</b> — Detect when speech is present</li>
    <li>🔀 <b>Overlap Detection</b> — Spot simultaneous speakers</li>
    <li>👥 <b>Speaker Diarization</b> — Auto-detect & label speakers</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
<div class="feature-card">
  <div class="feature-title">Understanding & Export</div>
  <ul class="feature-list">
    <li>🎭 <b>Emotion Recognition</b> — Track emotional tone</li>
    <li>🎙️ <b>Speech Transcription</b> — Convert speech to rich text</li>
    <li>💾 <b>Multiple Export Formats</b> — JSON, TXT, SRT, VTT</li>
  </ul>
</div>
""",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
