import streamlit as st

st.set_page_config(page_title="Scouting App", layout="wide")

st.title("⚽ Scouting & Similarity App")

st.sidebar.header("Select Position")

position = st.sidebar.radio(
    "Choose a player position:",
    [
        "Fullbacks",
        "Center Backs",
        "Midfielders",
        "Attacking Mids & Wingers",
        "Forwards"
    ]
)

if position == "Fullbacks":
    exec(open("app_fb.py").read())
elif position == "Center Backs":
    exec(open("app_cb.py").read())
elif position == "Midfielders":
    exec(open("app_mid.py").read())
elif position == "Attacking Mids & Wingers":
    exec(open("app_am.py").read())
elif position == "Forwards":
    exec(open("app_fw.py").read())
