import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("final_df_fw_with_clusters.csv")

meta_cols = ["Player", "Birthdate", "Age", "League", "Club", "Footed", "Nationality", "Position", "Rating", "Potential", "Minutes"]

# === FEATURE SET ===
fw_features = [
    'Goals', 'Assists', 'npxG + xAG',                 # Scoring and expected involvement
    'Shots on Target', 'Goals/Shot',                  # Efficiency and quality of finishing
    'Average Shot Distance',                          # Shot selection
    'Touches (Att Pen)', 'Touches (Att 3rd)',         # Involvement in danger areas
    'Progressive Carries',                            # Beating defenders and directness
    'Carries into Penalty Area',                      # Penetration
    'Pass Completion %', 'Live-ball Passes',          # Linking play
    'SCA (Shot)', 'SCA (Take-On)', 'SCA (Live-ball Pass)', # All shot creation channels
    'GCA (Shot)', 'GCA (Take-On)', 'GCA (Live-ball Pass)', # All goal creation channels
    'Through Balls', 'Crosses',                       # Direct creativity
    'Offsides',                                       # Movement behind defense
    'Fouls Drawn',                                    # Provoking fouls, winning set pieces
    'Aerials Won',                                    # Headers, target man play
    'Miscontrols', 'Dispossessed'                     # Ball retention under pressure
]

radar_features = [
    'Goals', 'npxG + xAG',                                          # Scoring and expected involvement
    'Goals/Shot', 'Average Shot Distance',                          # Shot selection
    'Touches (Att Pen)', 'Touches (Att 3rd)',                       # Involvement in danger areas
    'Carries into Penalty Area',                                    # Penetration
    'GCA (Shot)', 'SCA (Take-On)', 'SCA (Live-ball Pass)',          # All goal creation channels
    'Aerials Won',                                                  # Headers, target man play
]

cluster_names = {0: ('Aerial Target Men', 68),
 1: ('Hybrid Line Leaders', 24),
 2: ('Creative Withdrawn Forwards', 33),
 3: ('Traditional Finishers', 33),
 4: ('Physical Disruptors', 80),
 5: ('Dynamic Strike-Runners', 82)
 }

# Cluster descriptions
cluster_descriptions = {
    "Aerial Target Men": "This group is best described as \"Aerial Target Men\"—forwards who act as the vertical reference point in positional attacks. With players like Rodrigo Muniz, Ludovic Ajorque, Paul Onuachu, and Vedat Muriqi, this archetype thrives on physical dominance, aerial duels, and high-contact duels in the box. The group posts a modest 0.30 goals and 0.09 assists per 90, but compensates with extremely high aerial duel wins (4.80 per 90) and a Goals/Shot ratio (0.12) indicative of their poaching quality. Their average shot distance is the lowest of all clusters (12.93m), confirming their close-range role, typically after holding off defenders or getting on the end of crosses. With relatively low progressive carry numbers (0.83) and attacking third touches (14.3), these players aren't tasked with creation or combination play but are finishers through and through. Tactical Fit: These strikers are optimal in low-block or mid-block systems where build-up bypasses midfield via long balls or where sustained wide attacks create consistent crossing scenarios. A team lacking physicality in the final third can utilize them to secure territory and absorb pressure. Best fit in 4-2-3-1 or 3-5-2 systems alongside a more mobile second striker or attacking midfielder.",

    "Hybrid Line Leaders": "Cluster 1 presents a profile of \"Hybrid Line Leaders\", epitomized by Harry Kane, Kylian Mbappé, Julián Álvarez, and Viktor Gyökeres. These forwards combine scoring (0.54 goals/90) and creative output (0.27 assists/90) at a top-tier level, with a healthy xG+xAG of 0.66. They also lead all clusters in shots on target (1.33/90) while retaining strong Goals/Shot efficiency (0.13), suggesting shot selection isn't sacrificed for volume. These players operate across the entire attacking front—high in attacking third touches (23.1) and progressive carries (2.90), but equally capable in off-ball movement and transitions. Their high GCA (Goal-Creating Actions) from take-ons (0.10) and live-ball passes (0.39) implies the ability to beat players and create chances dynamically. Tactical Fit: These are universal forwards, adaptable across pressing, possession, and transitional systems. Whether leading the line solo in a 4-3-3 or pairing with a poacher in a 4-4-2 diamond, they can drop between lines to combine or threaten in behind. Ideal for systems requiring fluid interchanges and multifunctional attacking play, such as Manchester City, Arsenal, or Leipzig-style positional play.",

    "Creative Withdrawn Forwards": "This profile features \"Creative Withdrawn Forwards\", with prototypes such as Antoine Griezmann, Paulo Dybala, and Albert Guðmundsson. These players post lower goal outputs (0.23 goals/90) but excel in deeper link-up roles, as shown by touches in the attacking third (19.2), progressive carries (2.09), and live-ball chance creation (0.16 GCA/90). Their higher average shot distance (18.3m) underlines their tendency to shoot from the edge of the box rather than poaching in the area. They also contribute modestly to through balls and crosses, playing in between the lines rather than making penetrating runs. Tactical Fit: Best suited as false nines or second strikers in asymmetric frontlines. They excel in ball-dominant systems that rely on half-space occupation and intelligent movement. Ideal for a 4-4-2 diamond (at the tip), 3-4-2-1 setups, or as wide creators in narrow 4-2-3-1 formations. These forwards are not primary scorers but enhance collective attacking patterns through subtlety and intelligence.",

    "Traditional Finishers": "Cluster 3 gathers the \"Traditional Finishers\", a cohort of strikers with elite goal-scoring instincts and limited involvement in creation. Featuring Erling Haaland, Robert Lewandowski, Gonçalo Ramos, and Serhou Guirassy, these players top the group in goal output (0.74 goals/90) and non-penalty xG + xAG (0.76), with a robust 1.51 shots on target/90. They are low in carry volume (0.98 progressive carries) and rarely assist (0.07), reflecting a laser focus on end-product. They operate predominantly inside the box (average shot distance: 13.6m), thriving off final pass service rather than ball progression. Tactical Fit: These players demand high-volume chance creation systems and are most valuable when surrounded by playmakers. Ideal as the central pivot in a 4-2-3-1 or a front pairing in 3-5-2 with a more mobile second forward. They require tactical frameworks with width, cutbacks, and overloads to supply their finishing prowess.",

    "Physical Disruptors": "This group, characterized as \"Physical Disruptors\", includes Jamie Vardy, Álvaro Morata, Joshua Zirkzee, and Evanilson. Statistically, they present moderate goal output (0.30) and modest assist values (0.07), but are disruptive through high involvement in aerials, fouls drawn (1.77), and off-ball movements (offside calls at 0.66 per 90). They average the highest crosses attempted (2.74) among forwards, indicating frequent wide positioning or interchanging roles. Their miscontrol and dispossession numbers are also high, underlining a raw, combative nature rather than finesse. Tactical Fit: These forwards excel in high-tempo pressing systems where aggression, movement, and chaos are assets. Suitable for counter-pressing or transitional teams such as Leipzig or Brighton. They can play as wide forwards in 4-3-3, or mobile center-forwards in 4-2-3-1 systems where the emphasis is on disrupting defensive shapes and attacking space.",

    "Dynamic Strike-Runners": "Finally, we have the \"Dynamic Strike-Runners\", a modern breed headlined by Lautaro Martínez, Darwin Núñez, Marcus Thuram, and Ollie Watkins. These players are high-volume movers with excellent balance across goals (0.54), assists (0.15), and progressive actions (1.67 carries/90). They combine penalty-box instincts with the ability to stretch defenses. Posting strong shots on target (1.24) and touches in the attacking third (17.5), they embody the complete modern forward—able to lead transitions, engage defenders physically, and threaten in behind or on the ball. Tactical Fit: Best deployed in vertical, transition-heavy systems or hybrid possession models requiring pace and work-rate. Their capacity to press, run channels, and finish makes them ideal in 4-4-2 as split forwards or in narrow 4-3-3 setups. They can stretch backlines and attack space, serving as both scorers and pressure triggers."
}


# PCA setup
X = df[fw_features]
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
st.title("🎯 Forwards")

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
    st.dataframe(detailed_df[["Player", "Similarity"] + meta_cols[1:] + ["Cluster Name"] + fw_features].set_index("Player"))

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
    st.dataframe(df_filtered[meta_cols + ["Cluster Name"] + fw_features].set_index("Player"))
