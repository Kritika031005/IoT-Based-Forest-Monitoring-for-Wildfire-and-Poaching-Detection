import streamlit as st
import pandas as pd
import time
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="Forest Monitoring", layout="wide")

LOG_FILE = "forest_alerts_log.csv"

# ==========================
# HEADER
# ==========================
st.markdown("<h1 style='text-align:center;'>🌲 Forest Monitoring System</h1>", unsafe_allow_html=True)


# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙️ Control Panel")
refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 1, 10, 3)

st.sidebar.subheader("🔍 Filter Data")
filter_option = st.sidebar.selectbox(
    "Select Event Type",
    ["All", "Fire", "Poaching", "Logging"]
)

# ==========================
# LOAD DATA
# ==========================
def load_data():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return None

df = load_data()
# ==========================
# SESSION STATE (FOR ALERT TRACKING)
# ==========================
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

if df is None or df.empty:
    st.warning("⚠️ No data available yet. Run inference first.")
    st.stop()

# ==========================
# APPLY FILTER
# ==========================
if filter_option == "Fire":
    df = df[df["Fire Prediction"] == 2]
elif filter_option == "Poaching":
    df = df[df["Sound Prediction"] == "gunshot_clean"]
elif filter_option == "Logging":
    df = df[df["Sound Prediction"] == "logging_clean"]

# ==========================
# LATEST RECORD
# ==========================
latest = df.iloc[-1]

temp = latest["Temperature"]
pressure = latest["Pressure"]
humidity = latest["Humidity"]
label = latest["Sound Prediction"]
fire_pred = latest["Fire Prediction"]

# ==========================
# TOP METRICS
# ==========================
col1, col2, col3, col4 = st.columns(4)

col1.metric("🌡 Temp", f"{temp:.2f} °C")
col2.metric("📉 Pressure", f"{pressure:.2f} hPa")
col3.metric("💧 Humidity", f"{humidity:.2f} %")

if fire_pred == 2:
    col4.error("🔥 FIRE")
elif label in ["gunshot_clean", "logging_clean"]:
    col4.warning("⚠️ ALERT")
else:
    col4.success("🟢 NORMAL")

st.caption(f"Last Updated: {latest['Timestamp']}")
# ==========================
# LIVE POP-UP ALERT SYSTEM
# ==========================

current_alert = ""

if fire_pred == 2:
    current_alert = "🔥 FIRE DETECTED!"
elif label == "gunshot_clean":
    current_alert = "🚨 POACHING DETECTED!"
elif label == "logging_clean":
    current_alert = "🌲 ILLEGAL LOGGING DETECTED!"
else:
    current_alert = "NORMAL"

# Trigger popup only if new alert
if current_alert != st.session_state.last_alert:
    
    if current_alert != "NORMAL":
        st.toast(current_alert, icon="🚨")

        # Optional sound alert (browser beep)
        st.markdown(
            """
            <audio autoplay>
                <source src="https://www.soundjay.com/buttons/sounds/beep-01a.mp3" type="audio/mpeg">
            </audio>
            """,
            unsafe_allow_html=True
        )

    st.session_state.last_alert = current_alert

# ==========================
# TABS
# ==========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🚨 Alerts",
    "📍 Map",
    "🔊 Audio",
    "📜 History"
])

# ==========================
# TAB 1: OVERVIEW
# ==========================
with tab1:
    st.subheader("📈 Environmental Trends")

    fig_line = px.line(df, y=["Temperature", "Humidity"])
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("📊 Event Distribution")

    event_counts = {
        "Fire": len(df[df["Fire Prediction"] == 2]),
        "Poaching": len(df[df["Sound Prediction"] == "gunshot_clean"]),
        "Logging": len(df[df["Sound Prediction"] == "logging_clean"])
    }

    fig_pie = px.pie(
        names=list(event_counts.keys()),
        values=list(event_counts.values())
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================
# TAB 2: ALERTS
# ==========================
with tab2:
    st.subheader("🚨 Threat Detection")

    if label == "gunshot_clean":
        st.error("🚨 POACHING DETECTED")
    elif label == "logging_clean":
        st.error("🌲 ILLEGAL LOGGING DETECTED")
    elif fire_pred == 2:
        st.error("🔥 HIGH FIRE RISK")
    else:
        st.success("✅ NORMAL FOREST ACTIVITY")
    st.write("")  # spacing
    st.subheader("📊 Alert Summary")

    colA, colB, colC = st.columns(3)

    colA.metric("🔥 Fire Alerts", len(df[df["Fire Prediction"] == 2]))
    colB.metric("🔫 Poaching", len(df[df["Sound Prediction"] == "gunshot_clean"]))
    colC.metric("🌲 Logging", len(df[df["Sound Prediction"] == "logging_clean"]))

# ==========================
# TAB 3: MAP
# ==========================
with tab3:
    st.subheader("📍 Location Map")

    map_df = pd.DataFrame({
        "lat": [latest["Latitude"]],
        "lon": [latest["Longitude"]]
    })

    st.map(map_df)

# ==========================
# TAB 4: AUDIO
# ==========================
with tab4:
    st.subheader("🔊 Audio Analysis")

    AUDIO_FILE = "../scripts/recorded_audio.wav"

    if os.path.exists(AUDIO_FILE):
        st.audio(AUDIO_FILE)

        audio, sr = librosa.load(AUDIO_FILE, sr=16000)

        fig1, ax1 = plt.subplots()
        ax1.plot(audio[:2000])
        ax1.set_title("Waveform")
        st.pyplot(fig1)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        fig2, ax2 = plt.subplots()
        librosa.display.specshow(mfcc, ax=ax2)
        ax2.set_title("MFCC")
        st.pyplot(fig2)

    else:
        st.info("No audio file found")

# ==========================
# TAB 5: HISTORY
# ==========================

st.markdown("---")

with tab5:
    st.subheader("📜 Event History")

    def highlight(row):
        if row["Fire Prediction"] == 2:
            return ["background-color: red"] * len(row)
        elif row["Sound Prediction"] in ["gunshot_clean", "logging_clean"]:
            return ["background-color: orange"] * len(row)
        return [""] * len(row)

    styled_df = df.tail(20)

    def color_rows(row):
        if row["Fire Prediction"] == 2:
            return ["background-color: red"] * len(row)
        elif row["Sound Prediction"] in ["gunshot_clean", "logging_clean"]:
            return ["background-color: orange"] * len(row)
        return [""] * len(row)

    st.dataframe(
        styled_df.style.apply(color_rows, axis=1),
        height=400,
        use_container_width=True
    )

    st.subheader("📥 Download Logs")

    with open(LOG_FILE, "rb") as file:
        st.download_button(
            label="Download CSV",
            data=file,
            file_name="forest_logs.csv",
            mime="text/csv"
        )

# ==========================
# AUTO REFRESH
# ==========================
time.sleep(refresh_rate)
st.rerun()

