import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from services.auth.login_wall import render_login_wall, render_profile_card
from services.state.session_defaults import initial_session_defaults
from services.config.workout_config import EXERCISE_OPTIONS
from services.ui.style_loader import load_css, inject_local_font, inject_webrtc_styles, apply_theme
from services.persistence.exercise_repository import init_db
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from services.vision.exercise_video_processor import VideoProcessorClass
from services.tracking.metrics import sync_metrics_update
from services.persistence.exercise_repository import get_users_exercises, add_exercise
from groq import Groq
from services.coaching.llm import LLMCoach
from services.coaching.tts import TextToSpeech
from services.coaching.voice_pipeline import VoicePipeline, autoplay_audio
 
EXERCISE_META = {
    "Squats":                  {"icon": "🦵", "color": "#6C63FF", "desc": "Legs & Glutes"},
    "Push-ups":                {"icon": "💪", "color": "#22C55E", "desc": "Chest & Triceps"},
    "Biceps Curls (Dumbbell)": {"icon": "🏋️", "color": "#F59E0B", "desc": "Biceps & Arms"},
    "Shoulder Press":          {"icon": "🙌", "color": "#EF4444", "desc": "Shoulders & Traps"},
    "Lunges":                  {"icon": "🚶", "color": "#06B6D4", "desc": "Legs & Balance"},
}
 
 
def render_exercise_cards(selected):
    cols = st.columns(len(EXERCISE_META))
    for i, (name, meta) in enumerate(EXERCISE_META.items()):
        active = name == selected
        border = meta["color"] if active else "var(--border-color)"
        bg = f"rgba({','.join(str(int(meta['color'].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.12)" if active else "var(--card-bg)"
        cols[i].markdown(
            f"""<div style="
                border:2px solid {border};
                border-radius:12px;
                padding:14px 8px;
                text-align:center;
                background:{bg};
                transition:all 0.2s;
                cursor:default;
            ">
                <div style="font-size:1.8rem">{meta['icon']}</div>
                <div style="font-size:11px;font-weight:700;margin-top:6px;color:var(--text-color)">{name.split('(')[0].strip()}</div>
                <div style="font-size:10px;opacity:0.6;color:var(--text-color)">{meta['desc']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
 
 
def render_stats_summary(df):
    total_sessions = len(df.groupby(["Exercise", "Date"]))
    total_reps     = int(df["Reps"].sum())
    total_sets     = int(df["Sets"].sum())
    total_min      = int(df["Time (sec)"].sum() // 60)
    fav_exercise   = df.groupby("Exercise")["Reps"].sum().idxmax() if not df.empty else "—"
    fav_meta       = EXERCISE_META.get(fav_exercise, {"icon": "🏅", "color": "#6C63FF"})
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🗓️ Sessions",    total_sessions)
    c2.metric("🔁 Total Reps",  total_reps)
    c3.metric("📦 Total Sets",  total_sets)
    c4.metric("⏱️ Minutes",     total_min)
 
    st.markdown(
        f"""<div style="
            margin:12px 0 0 0;
            padding:10px 16px;
            border-radius:10px;
            background:rgba({','.join(str(int(fav_meta['color'].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.12);
            border:1px solid {fav_meta['color']};
            font-size:14px;
            color:var(--text-color);
        ">
            {fav_meta['icon']} <strong>Favourite Exercise:</strong> {fav_exercise}
        </div>""",
        unsafe_allow_html=True,
    )
 
 
def render_history_chart(df, theme):
    agg = df.groupby(["Exercise", "Date"]).agg({"Reps": "sum", "Sets": "sum", "Time (sec)": "sum"}).reset_index()
 
    tab1, tab2, tab3 = st.tabs(["📊 Reps Over Time", "📦 Sets by Exercise", "⏱️ Time Spent"])
 
    plot_bg   = "rgba(0,0,0,0)"
    paper_bg  = "rgba(0,0,0,0)"
    font_col  = "#ffffff" if theme == "dark" else "#1A1D2E"
    grid_col  = "rgba(255,255,255,0.07)" if theme == "dark" else "rgba(0,0,0,0.07)"
 
    base_layout = dict(
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(color=font_col, family="sans-serif"),
        xaxis=dict(gridcolor=grid_col, showgrid=True),
        yaxis=dict(gridcolor=grid_col, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
 
    colors = [m["color"] for m in EXERCISE_META.values()]
 
    with tab1:
        fig1 = px.line(
            agg, x="Date", y="Reps", color="Exercise",
            markers=True, color_discrete_sequence=colors,
        )
        fig1.update_layout(**base_layout)
        fig1.update_traces(line_width=2.5, marker_size=7)
        st.plotly_chart(fig1, use_container_width=True)
 
    with tab2:
        sets_by_ex = agg.groupby("Exercise")["Sets"].sum().reset_index()
        fig2 = px.bar(
            sets_by_ex, x="Exercise", y="Sets",
            color="Exercise", color_discrete_sequence=colors,
            text="Sets",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(**base_layout, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
 
    with tab3:
        agg["Minutes"] = (agg["Time (sec)"] / 60).round(1)
        time_by_ex = agg.groupby("Exercise")["Minutes"].sum().reset_index()
        fig3 = px.pie(
            time_by_ex, names="Exercise", values="Minutes",
            color="Exercise", color_discrete_sequence=colors,
            hole=0.45,
        )
        fig3.update_layout(**base_layout)
        fig3.update_traces(textfont_color=font_col)
        st.plotly_chart(fig3, use_container_width=True)
 
 
def main():
    load_dotenv()
 
    st.set_page_config(
        page_icon="🤸",
        page_title="PoseCoach AI",
        initial_sidebar_state="expanded",
        layout="centered"
    )
 
    # Theme state — must be before CSS inject
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
 
    theme = st.session_state.get("theme", "dark")
 
    load_css(os.path.join(os.getcwd(), "static", "style.css"))
    inject_local_font(os.path.join(os.getcwd(), "static", "AdobeClean.otf"), "AdobeClean")
    apply_theme(theme)
 
    init_db()
 
    if not render_login_wall():
        return
 
    initial_session_defaults()
 
    if "voice_pipeline" not in st.session_state:
        try:
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key and hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
            if not api_key or not str(api_key).strip():
                raise ValueError("GROQ_API_KEY is missing. Add it to .env or Streamlit secrets.")
            groq_client = Groq(api_key=api_key)
            llm_coach = LLMCoach(groq_client)
            tts = TextToSpeech()
            st.session_state.voice_pipeline = VoicePipeline(llm_coach, tts)
        except Exception as e:
            st.session_state.voice_pipeline = None
            st.session_state.voice_error = str(e)
 
    workout_started = st.session_state.get("workout_started", False)
 
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("🤸 PoseCoach AI")
        with col2:
            icon = "☀️" if theme == "dark" else "🌙"
            label = "Light" if theme == "dark" else "Dark"
            if st.button(icon, key="theme_toggle", help=f"Switch to {label} mode"):
                st.session_state["theme"] = "light" if st.session_state.get("theme", "dark") == "dark" else "dark"
                st.rerun()
 
        render_profile_card()
        st.divider()
 
        if st.session_state.get("voice_error"):
            st.warning(f"Voice feedback disabled: {st.session_state.voice_error}")
 
        st.subheader("Workout Plan")
 
        if not workout_started:
            plan_exercise = st.selectbox("Exercise", options=EXERCISE_OPTIONS, key="plan_exercise")
            plan_sets = st.number_input("Sets", min_value=0, max_value=50, key="plan_sets", step=1)
            plan_reps = st.number_input("Reps per Set", min_value=0, max_value=50, key="plan_reps", step=1)
            st.markdown("")
 
            start_session_button = st.button("Start Workout", width="stretch", key="start_session_button")
 
            if start_session_button:
                st.session_state.exercise_type = plan_exercise
                st.session_state.target_sets = int(plan_sets)
                st.session_state.reps_per_set = int(plan_reps)
                st.session_state.reps = 0
                st.session_state.workout_started = True
                st.session_state.set_cycle_started_at = time.time()
                st.session_state.last_saved_sets_completed = 0
 
                if st.session_state.voice_pipeline:
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_started", exercise=plan_exercise, metrics={}
                    )
                    if result:
                        st.session_state.audio_to_play, st.session_state.coach_feedback = result
 
                st.session_state.last_notified_sets_completed = 0
                st.session_state.last_notified_workout_complete = False
                st.rerun()
        else:
            exercise = st.session_state.get("exercise_type")
            sets = st.session_state.get("target_sets")
            reps = st.session_state.get("reps_per_set")
            meta = EXERCISE_META.get(exercise, {"icon": "🏋️"})
            st.info(f"{meta['icon']} **{exercise}** — {sets} Sets / {reps} Reps")
 
            end_session_button = st.button("End Workout", key="end_session_button", width="stretch")
 
            if end_session_button:
                st.session_state.workout_started = False
                user_id = st.session_state.get("user_id", 0)
                total_reps = st.session_state.get("reps", 0)
                sets_completed = st.session_state.get("sets_completed", 0)
                elapsed_time = int(time.time() - st.session_state.get("set_cycle_started_at", time.time()))
                if isinstance(user_id, int) and (total_reps > 0 or sets_completed > 0):
                    add_exercise(user_id, exercise, total_reps, sets_completed, elapsed_time)
                if st.session_state.voice_pipeline:
                    result = st.session_state.voice_pipeline.process_event(
                        event="workout_completed", exercise=exercise, metrics={}
                    )
                    if result:
                        st.session_state.audio_to_play, st.session_state.coach_feedback = result
                st.rerun()
 
        if workout_started:
            st.divider()
            exercise     = st.session_state.get("exercise_type")
            total_reps   = st.session_state.get("reps")
            current_set_reps = st.session_state.get("current_set_reps")
            reps_per_set = st.session_state.get("reps_per_set")
            sets_completed = st.session_state.get("sets_completed")
            target_sets  = st.session_state.get("target_sets")
 
            st.subheader("Progress")
            st.metric("Total Reps", f"{total_reps}")
            st.metric("Current Set Reps", f"{current_set_reps} / {reps_per_set}")
            st.metric("Sets Completed", f"{sets_completed} / {target_sets}")
 
            st.divider()
            st.subheader("Form Quality")
            form_score = st.session_state.get("form_score", 0)
            badge = "🟢" if form_score >= 80 else ("🟡" if form_score >= 50 else "🔴")
            st.metric("Form Score", f"{badge} {form_score} / 100")
 
            form_feedbacks = st.session_state.get("form_feedbacks", [])
            for fb in form_feedbacks:
                st.markdown(fb)
 
            st.divider()
 
            if exercise == "Squats":
                st.subheader("Squat Metrics")
                st.metric("Knee Angle", f"{st.session_state.knee_angle}°")
                st.metric("Back Angle", f"{st.session_state.back_angle}°")
                st.metric("Depth Status", st.session_state.depth_status)
            elif exercise == "Push-ups":
                st.subheader("Push-up Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Body Alignment", st.session_state.body_alignment)
                st.metric("Hip Position", st.session_state.hip_status)
            elif exercise == "Biceps Curls (Dumbbell)":
                st.subheader("Curl Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Shoulder Stability", st.session_state.shoulder_status)
                st.metric("Swing Detection", st.session_state.swing_status)
            elif exercise == "Shoulder Press":
                st.subheader("Shoulder Press Metrics")
                st.metric("Elbow Angle", f"{st.session_state.elbow_angle}°")
                st.metric("Arm Extension", st.session_state.extension_status)
                st.metric("Back Arch", st.session_state.back_arch_status)
            elif exercise == "Lunges":
                st.subheader("Lunge Metrics")
                st.metric("Front Knee Angle", f"{st.session_state.front_knee_angle}°")
                st.metric("Torso Angle", f"{st.session_state.torso_angle}°")
                st.metric("Balance Status", st.session_state.balance_status)
 
    # ── MAIN CONTENT ──────────────────────────────────────────────────
    st.title("🤸 PoseCoach AI")
    st.markdown("#### Real-time pose detection with proactive AI voice coaching")
 
    if st.session_state.get("audio_to_play"):
        autoplay_audio(st.session_state.audio_to_play)
        st.session_state.audio_to_play = None
 
    if st.session_state.get("coach_feedback"):
        st.markdown("")
        st.success(f"🤖 **Coach:** {st.session_state.coach_feedback}")
 
    # Exercise cards (always visible)
    selected_exercise = st.session_state.get("exercise_type") or st.session_state.get("plan_exercise", EXERCISE_OPTIONS[0])
    render_exercise_cards(selected_exercise)
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
 
    if not workout_started:
        border_col = "var(--border-color, #444)"
        text_col   = "var(--text-color, #ccc)"
        st.markdown(
            f"""<div style="
                border: 3px dashed {border_col};
                border-radius: 14px;
                padding: 48px 32px;
                text-align: center;
                margin-top: 16px;
                margin-bottom: 32px;
            ">
                <h2 style="color:{text_col};margin-bottom:8px;">👈 Set your workout plan</h2>
                <p style="font-size:1.05rem;color:{text_col};opacity:0.7;">
                    Choose your exercise, sets and reps in the sidebar,<br>
                    then click <strong>Start Workout</strong> to activate the camera and AI coach.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        import os
        is_cloud = os.environ.get("HOME", "").startswith("/home/adminuser")
        if is_cloud:
            st.info("Camera-based pose detection works when running locally. You can still save workouts using End Workout and view your history below.")
            st.markdown(
                "<div style='text-align:center;padding:40px;border:2px dashed #444;border-radius:14px;margin:16px 0'>"
                "<div style='font-size:3rem'>📷</div>"
                "<h3>Camera unavailable on Cloud</h3>"
                "<p>Run locally for full pose detection.<br>Use <strong>End Workout</strong> to save your session.</p>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            context = webrtc_streamer(
                key="exercise-analysis",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessorClass,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            sync_metrics_update(context)
            if context.state.playing:
                time.sleep(0.25)
                st.rerun()
            inject_webrtc_styles()
 
    # ── WORKOUT HISTORY ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Workout Dashboard")
 
    user_id = st.session_state.get("user_id", 0)
 
    if isinstance(user_id, int):
        history_rows = get_users_exercises(user_id)
        arr = [
            {
                "Exercise": row['exercise_name'],
                "Reps":     row['reps'],
                "Sets":     row['sets'],
                "Time (sec)": row['time'],
                "Date":     row['created_at']
            }
            for row in history_rows
        ]
        df = pd.DataFrame(arr)
 
        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
 
            # Stats summary
            render_stats_summary(df)
            st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
 
            # Charts
            render_history_chart(df, theme)
 
            # Raw table (collapsible)
            with st.expander("📋 View Raw History"):
                agg_df = df.groupby(["Exercise", "Date"]).agg({
                    "Reps": 'sum', "Sets": "sum", "Time (sec)": "sum"
                }).reset_index()
                agg_df.index += 1
                st.dataframe(agg_df, use_container_width=True)
        else:
            st.info("No workout history found. Complete a workout to see your stats here!")
 
 
if __name__ == "__main__":
    main()


