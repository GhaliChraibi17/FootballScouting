import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("final_df_am_with_clusters.csv")

meta_cols = ["Player", "Birthdate", "Age", "League", "Club", "Footed", "Nationality", "Position", "Rating", "Potential", "Minutes"]

# === FEATURE SET ===
am_features = [
    'Goals', 'Assists', 'npxG + xAG', 'Shots on Target', 'Goals/Shot', 'Average Shot Distance',
    'Progressive Carries', 'Progressive Carrying Distance',
    'Carries into Final Third', 'Carries into Penalty Area',
    'Successful Take-Ons', 'Successful Take-On %', 'Touches (Att 3rd)', 'Touches (Att Pen)',
    'Pass Completion %', 'Live-ball Passes', 'Progressive Passes', 'Progressive Passing Distance',
    'Passes into Final Third', 'Passes into Penalty Area', 'Crosses', 'Crosses into Penalty Area',
    'Through Balls', 'Switches', 'SCA (Live-ball Pass)', 'SCA (Take-On)', 'SCA (Shot)',
    'GCA (Live-ball Pass)', 'GCA (Take-On)', 'GCA (Shot)', 'Miscontrols', 'Dispossessed', 'Fouls Drawn'
]

radar_features = [
    "Goals", "Assists", "npxG + xAG", "Progressive Carries",
    "Carries into Final Third", "Carries into Penalty Area",
    "SCA (Live-ball Pass)", "SCA (Take-On)",
    'Passes into Final Third', 'Passes into Penalty Area', 'Successful Take-Ons'
]

cluster_names = {
    0: ("Hybrid Orchestrators", 121),
    1: ("Direct Dribbling Threats", 164),
    2: ("Secondary Attackers", 171),
    3: ("Elite Technicians", 46),
    4: ("World-Class Wingers", 75)
}

# Cluster descriptions
cluster_descriptions = {
    "Hybrid Orchestrators": "These are players who thrive between the lines, acting as intelligent connectors between midfield and attack. Despite modest goal (0.19) and assist (0.19) outputs, their value lies in maintaining rhythm and exploiting micro-spaces. Their average shot distance (19.6m) suggests low central box occupation—opting instead for late arrivals or second-ball strikes.\n\nThey average 2.29 progressive carries and 78m in carrying distance, indicating a preference for subtle, tempo-driven advances rather than explosive dribbles. High \"live-ball pass SCA\" (2.57) and modest take-on creation reflect a reliance on structured possession.\n\nTactical Fit:\nIdeal for a 3-2-4-1 or 4-3-3 where positional rotations are essential. Think of them as the “Iniesta or Silva\" roles: not flamboyant, but crucial for synchronized ball movement.",
    "Direct Dribbling Threats": "This cluster contains wide players with strong vertical thrust and final-third ambition. While goal output (0.23) is average, their shots on target per game (0.82) and progressive carrying distance (99m) showcase relentless ball progression.\n\nCrucially, they rank highest in take-on chance creation (0.36) and miscontrols (2.65), a statistical mark of risk-taking wingers who seek isolation duels. Carries into the final third (2.15) and switch plays are frequent, making them ideal for destabilizing defensive blocks.\n\nTactical Fit:\nFlourish in 4-2-3-1 or 4-3-3 setups that value wide penetration. Their role resembles Leroy Sané or Raheem Sterling: wing scorers with freedom to drive inside.",
    "Secondary Attackers": "This group shows subdued final-third productivity (0.18 goals, 0.11 assists) but brings tactical discipline, vertical ball progression and defensive presence. They register 1.94 progressive carries (modest) and lower creative stats but compensate with intelligent movement and support play.\n\nFinal third carries (1.23) and low take-on contribution highlight a direct, no-nonsense profile. This is your pressing ten or inside midfielder who balances risk and retention.\n\nTactical Fit:\nPerfect as interior midfielders in 4-4-2 diamond or 3-4-2-1—trusted to shuttle, press, and deliver basic progression.",
    "Elite Technicians": "This cluster blends elite-level output (0.30 goals, 0.27 assists) with orchestration. They average 0.82 shots on target, 3.10 progressive carries, and are the top cluster for total creative actions: Live-ball SCA: 3.54, Live-ball GCA: 0.42.\n\nThey're high-volume, high-impact creators who don’t merely support attacks—they initiate and end them. Their average shot distance is longest (20.05m), indicating confident shooters from range.\n\nTactical Fit:\nThese players are tactical nuclei in 3-2-2-3 or 4-2-3-1, where all final-third orchestration runs through them.",
    "World-Class Wingers": "This cluster houses elite wingers and attacking midfielders with superior output across the board: Goals: 0.36, Assists: 0.26, Shots on target: 1.02, npxG+xAG: 0.56, Take-on SCA: 0.53 (highest).\n\nThey’re both creators and finishers—true reference points. With the highest progressive carries (5.08) and 135m carrying distance, they dominate wide zones or cut-inside lanes. They’re also the most frequently fouled, and least dispossessed.\n\nTactical Fit:\nIn a 3-2-5 or 2-3-5, these are your wide apexes. They pin full-backs, create gravity zones, and score double digits consistently. They don’t just fit into systems—they define them."
}


# PCA setup
X = df[am_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
df_pca = X_pca
cluster_labels = df["Cluster"].values
player_index = {player: idx for idx, player in enumerate(df["Player"])}
lower_bounds = df[radar_features].quantile(0.02)
upper_bounds = df[radar_features].quantile(0.98)
range_vals = (upper_bounds - lower_bounds).replace(0, 1)

# Similarity function
def compute_similarity(player_name, df_pca, cluster_labels, player_index, boost=1.1):
    base_idx = player_index[player_name]
    base_vec = df_pca[base_idx]
    similarities = []
    for name, idx in player_index.items():
        if name == player_name:
            continue
        other_vec = df_pca[idx]
        sim = cosine_similarity([base_vec], [other_vec])[0, 0] * 100
        if cluster_labels[idx] == cluster_labels[base_idx]:
            sim *= boost
        similarities.append((name, round(min(sim, 100), 2)))
    return sorted(similarities, key=lambda x: -x[1])

# Score function
def similarity_score(p1, p2, df_pca, cluster_labels, player_index, boost=1.1):
    idx1, idx2 = player_index[p1], player_index[p2]
    sim = cosine_similarity([df_pca[idx1]], [df_pca[idx2]])[0, 0] * 100
    if cluster_labels[idx1] == cluster_labels[idx2]:
        sim *= boost
    return round(min(sim, 100), 2)

def plot_radar(df, p1, p2, features, lb, ub):
    p1_raw, p2_raw = df[df["Player"] == p1][features].values[0], df[df["Player"] == p2][features].values[0]
    range_vals = (ub - lb).replace(0, 1)
    p1_scaled = ((p1_raw - lb) / range_vals).clip(0, 1)
    p2_scaled = ((p2_raw - lb) / range_vals).clip(0, 1)

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0]
    p1_scaled, p2_scaled = np.append(p1_scaled, p1_scaled[0]), np.append(p2_scaled, p2_scaled[0])
    p1_raw, p2_raw = np.append(p1_raw, p1_raw[0]), np.append(p2_raw, p2_raw[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, p1_scaled, color="green", linewidth=2, label=p1)
    ax.fill(angles, p1_scaled, color="green", alpha=0.25)
    ax.plot(angles, p2_scaled, color="red", linewidth=2, label=p2)
    ax.fill(angles, p2_scaled, color="red", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=8)
    ax.set_yticks(np.linspace(0, 1, 10))
    ax.set_yticklabels([])
    for angle, val1, val2, scale1, scale2 in zip(angles[:-1], p1_raw[:-1], p2_raw[:-1], p1_scaled[:-1], p2_scaled[:-1]):
        ax.text(angle, scale1 * 0.9, f"{val1:.2f}", ha='center', va='center', fontsize=7, color='green')
        ax.text(angle, scale2 * 0.8, f"{val2:.2f}", ha='center', va='center', fontsize=7, color='red')
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)
    st.pyplot(fig)

def create_cluster_radar():
    raw_means = df.groupby("Cluster")[radar_features].mean()
    range_vals = (upper_bounds - lower_bounds).replace(0, 1)
    scaled_means = (raw_means - lower_bounds) / range_vals
    scaled_means = scaled_means.clip(0, 1)
    
    cols = st.columns(3)
    for i, cluster_id in enumerate(raw_means.index):
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        stats = scaled_means.loc[cluster_id].values
        raw_stats = raw_means.loc[cluster_id].values
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist() + [0]
        stats = np.concatenate((stats, [stats[0]]))
        raw_stats = np.concatenate((raw_stats, [raw_stats[0]]))
        ax.plot(angles, stats, linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_features, fontsize=8)
        ax.set_yticks(np.linspace(0, 1, 10))
        ax.set_yticklabels([])
        for angle, scaled_val, raw_val in zip(angles[:-1], stats[:-1], raw_stats[:-1]):
            ax.text(angle, scaled_val * 0.8, f"{raw_val:.2f}", ha='center', va='center', fontsize=6)
        title, size = cluster_names[cluster_id]
        ax.set_title(f"{title} (n={size})", fontsize=11, y=1.1)
        cols[i % 3].pyplot(fig)


# === Streamlit App ===
st.title("🎯 Attacking Midfielders & Wingers")

page = st.sidebar.radio("Navigate", [
    "📌 Similarity Search", "🆚 Compare Players", "🧬 Cluster Profiles", "📊 Full Player Table"
])

if page == "📌 Similarity Search":
    player = st.selectbox("Select Player", df["Player"].unique())
    n = st.slider("Top N", 3, 30, 10)

    sim_list = compute_similarity(player, df_pca, cluster_labels, player_index)
    df_sim = pd.DataFrame(sim_list, columns=["Player", "Similarity"])

    top_players_df = df_sim.head(n)
    top_players = top_players_df["Player"].tolist()

    # Merge similarity scores into main DataFrame
    detailed_df = df[df["Player"].isin(top_players)].copy()
    detailed_df = detailed_df.merge(top_players_df, on="Player")
    detailed_df = detailed_df.sort_values("Similarity", ascending=False)

    st.subheader("🧠 Similar Players with Full Stats")
    st.dataframe(detailed_df[["Player", "Similarity"] + meta_cols[1:] + ["Cluster Name"] + am_features].set_index("Player"))

elif page == "🆚 Compare Players":
    p1 = st.selectbox("Player 1", df["Player"].unique())
    p2 = st.selectbox("Player 2", df["Player"].unique(), index=1)
    if st.button("Compare"):
        score = similarity_score(p1, p2, df_pca, cluster_labels, player_index)
        st.subheader(f"Similarity Score: {abs(score):.2f}")
        plot_radar(df, p1, p2, radar_features, lower_bounds, upper_bounds)

elif page == "🧬 Cluster Profiles":
    st.header("🧬 Cluster Spider Charts & Descriptions")
    st.markdown("Visual & tactical breakdown of each attacking midfielder/wide profile.")
    
    create_cluster_radar()
    st.markdown("---")

    for cname, cdesc in cluster_descriptions.items():
        st.markdown(f"""
        <div style='margin-bottom: 2rem;'>
            <h3 style='margin-bottom: 0.5rem; font-size:20px; color: #1E90FF;'>🔹 {cname}</h3>
            <div style='font-size: 16px; line-height: 1.6; color: white;'>{cdesc}</div>
        </div>
        <hr style='margin-top: 2rem; margin-bottom: 2rem;'>
        """, unsafe_allow_html=True)

elif page == "📊 Full Player Table":
    league = st.multiselect("League", df["League"].unique())
    age_slider = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (18, 32))
    club = st.multiselect("Club", df["Club"].unique())
    cluster = st.multiselect("Cluster Name", df["Cluster Name"].unique())
    foot = st.multiselect("Footed", df["Footed"].unique())
    nat = st.multiselect("Nationality", df["Nationality"].unique())

    df_filtered = df.copy()
    if league:
        df_filtered = df_filtered[df_filtered["League"].isin(league)]
    if club:
        df_filtered = df_filtered[df_filtered["Club"].isin(club)]
    if cluster:
        df_filtered = df_filtered[df_filtered["Cluster Name"].isin(cluster)]
    if foot:
        df_filtered = df_filtered[df_filtered["Footed"].isin(foot)]
    if nat:
        df_filtered = df_filtered[df_filtered["Nationality"].isin(nat)]
    if "Age" in df.columns:
        df_filtered = df_filtered[
            (df_filtered["Age"] >= age_slider[0]) & (df_filtered["Age"] <= age_slider[1])
        ]

    # --- Sort Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        sort_by_rating = st.button("🔝 Sort by Rating")
    with col2:
        sort_by_potential = st.button("🚀 Sort by Potential")

    if sort_by_rating:
        df_filtered = df_filtered.sort_values("Rating", ascending=False)
    elif sort_by_potential:
        df_filtered = df_filtered.sort_values("Potential", ascending=False)

    # Display
    st.dataframe(df_filtered[meta_cols + ["Cluster Name"] + am_features].set_index("Player"))
