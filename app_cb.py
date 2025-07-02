import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("final_df_cb_with_clusters.csv")

meta_cols = ["Player", "Birthdate", "Age", "League", "Club", "Footed", "Nationality", "Position", "Rating", "Potential", "Minutes"]

# === FEATURE SET ===
center_back_features = [
    'Goals', 'Assists',                                # Set piece threat
    'Pass Completion %', 'Pass Completion % (Long)',   # Build-up reliability
    'Progressive Passes', 'Progressive Passing Distance', # Line-breaking passes
    'Passes Attempted (Long)',                         # Direct balls out of defense
    'Blocks', 'Shots Blocked',                         # Shot prevention
    'Tackles (Def 3rd)', 'Tackles (Mid 3rd)',          # Defensive duels in own/mid half
    'Tkl+Int',                                         # Ball-winning
    'Aerials Won',                                     # Dominance in the air
    'Ball Recoveries',                                 # Sweeping up behind line
    'Fouls Committed', 'Yellow Cards', 'Red Cards',    # Discipline and aggression
    'Touches (Def 3rd)',                               # Involvement in deep build-up
    'GCA (Defensive Action)',                          # Direct defensive goal involvement
    'SCA (Defensive Action)'                          # Direct defensive shot prevention
]

radar_features = [                              
    'npxG + xAG',
    'Passes Attempted (Long)', 'Pass Completion % (Long)',   # Build-up reliability
    'Progressive Passes', 'Progressive Passing Distance',                    # Line-breaking passes
    'Blocks',                                          # Shot prevention
    'Tkl+Int',                                         # Ball-winning
    'Aerials Won',                                     # Dominance in the air
    'Ball Recoveries',                                 # Sweeping up behind line
    'Fouls Committed',                                 # Discipline and aggression
    'Touches (Def 3rd)'                                 # Involvement in deep build-up
    ]   

cluster_names = {0: ('Dominant Anchors', 83),
 1: ('Mobile Front-Foot Defenders', 20),
 2: ('Classical Defenders', 62),
 3: ('Balanced Ball-Players', 49),
 4: ('Elite Circulators', 74),
 5: ('Versatile Press Breakers', 28),
 6: ('No-Nonsense Guardians', 82),
 7: ('World-Class Hybrid Leaders', 51)}

# Cluster descriptions
cluster_descriptions = {
    "Dominant Anchors": "This group consists of physically commanding center backs who impose themselves through aerial duels and penalty box presence. Players like Jannik Vestergaard and José María Giménez exemplify this category—combining above-average aerial wins (2.31), significant long pass involvement (9.55 per 90), and an assertive defensive presence with nearly 3 defensive actions per game (Tackles + Interceptions: 2.66). While their progressive carrying and passing volumes are relatively average, their centrality comes through in possession when launching longer distributions or covering deep zones. Their defensive third touches (37.1 per 90) confirm a role deeply embedded in low blocks. These are traditional stoppers who provide security over flair.\n\nTactical Fit: These players are tailored for deep defensive systems such as a compact 4-4-2 or a back-three with high central congestion. Their strengths lie in protecting the box and clearing second balls. Ideal in teams playing reactively or in leagues where direct football is dominant.",

    "Mobile Front-Foot Defenders": "This cluster is headlined by players like Marquinhos and Ibrahima Konaté, combining mobile defending with forward-thinking instincts. Statistically, they balance strong progressive volume (3.88 progressive passes and ~339m progressive distance) with solid defensive output (2.67 tackles + interceptions). Their aerial success (2 per 90) is decent but not elite, hinting at profiles that favor recovery runs and anticipatory defense over physical dominance. They also possess one of the higher assist rates among clusters (0.0425), reinforcing the idea of defenders who step into midfield lines or break structure. Their foul rate is modest, and they exhibit intelligent timing with relatively low card incidence.\n\nTactical Fit: Suited for high defensive lines and systems that require proactive center backs—think Liverpool’s 4-3-3 or PSG’s hybrid back four. These defenders shine in possession-heavy teams needing speed in defensive transitions and positional flexibility.",

    "Classical Defenders": "Cluster 2 gathers robust defenders who engage frequently but with conservative ball usage. With relatively low pass completion (77.5%) and long-ball reliance (~7.8 attempts per 90), these players like James Tarkowski and Gustavo Gómez are more reactive than proactive. Their defensive volume is high (3.15 tackles + interceptions), and aerial duels per 90 are near elite (2.99), indicating their strength in traditional duel-heavy settings. Touches in the defensive third are the lowest across all clusters (27.4), showing they operate in systems where the center back is rarely a ball progression outlet. They are among the more fouled-prone defenders (1.15 fouls per 90) and receive the most yellow cards, reflecting their rugged style.\n\nTactical Fit: Perfect for mid- or low-block defenses, especially in teams fighting relegation or operating with man-marking principles. A solid fit in back-fours where the center back is asked to win first balls and not initiate play.",

    "Balanced Ball-Players": "This group represents center backs who bring a mix of technical security and defensive contribution. Featuring profiles like Lisandro Martínez and Willian Pacho, they maintain high pass accuracy (85.7%), mid-range progressive output (3.7 passes and ~299m distance), and a balanced aerial contribution (1.79 won per 90). Defensively, their volume is high (3.47 tackles + interceptions), and their touches indicate a steady involvement in buildup. Their foul count is stable, and their recovery actions suggest good reading of the game. These are modern center backs, capable of engaging but also guiding buildup through secure short and medium-range passing.\n\nTactical Fit: A natural fit in positional play structures such as 3-2-5 or 4-2-3-1 formations with emphasis on controlled buildup. These players are trusted to play through pressure, step into midfield, or split wide when needed.",

    "Elite Circulators": "This cluster contains the most secure passers of all, boasting the highest pass completion (87.8%) and long ball accuracy (61.6%). Names like Rúben Dias and Benjamin Pavard dominate here. With moderate progressive passing numbers and high defensive reliability (Tkl+Int: 2.14), they represent elite “stabilizers” in possession-centric teams. They don’t register high defensive actions because they operate in structures that dominate territory. Their recoveries (3.5) and touches (30.1) are evidence of their constant involvement in buildup and spatial control.\n\nTactical Fit: Ideal for teams that monopolize the ball, such as Manchester City or Inter Milan. They thrive in systems where defensive actions are preventative, and the main role of the center back is to orchestrate passing chains and maintain structural discipline.",

    "Versatile Press Breakers": "Players in this cluster (e.g., Militão, Fabian Schär, Salisu) are defined by their press resistance and verticality. They show the highest progressive pass volume (4.26 per 90) and a healthy long passing rate, yet with slightly less polish in pass accuracy (84.7%). Their defensive involvement remains solid (3.01 tackles + interceptions), and their touches profile them as key figures in transition setups. They're not pure destroyers nor metronomes, but hybrid center backs who often carry or pass through pressure. Their yellow card incidence is among the highest, indicating an aggressive engagement profile.\n\nTactical Fit: Best used in high-risk, high-reward setups such as pressing 3-4-3 or counter-pressing 4-2-3-1. These players can initiate attacks under pressure and recover aggressively when possession is lost.",

    "No-Nonsense Guardians": "This archetype is rooted in simplicity and assertiveness. Players like Wout Faes and Conor Coady form a block of old-school defenders. Their long-ball usage is moderately high (~7.2 per 90), but their progressive volume is the lowest in the dataset (2.21 passes, 282m). However, they excel in raw defensive stats: strong aerial presence (2.63 won per 90), decent recoveries, and above-average shot blocking. They foul less than Cluster 2, and their card count is among the most contained, reflecting experienced timing over sheer aggression.\n\nTactical Fit: Suited for transitional Premier League sides or Championship-level systems emphasizing physical duels and set-piece security. Typically deployed in compact 4-4-2s or flat 5-back systems.",

    "World-Class Hybrid Leaders": "This is the elite tier, featuring names like Virgil van Dijk, William Saliba, and Alessandro Bastoni. They combine best-in-class pass completion (90.5%), elite long pass accuracy (69.3%), and unrivaled defensive composure. Progressive metrics are top-tier (5.67 passes and 470m per 90), suggesting they are engines of buildup and diagonal progression. Their aerials and recoveries are robust, fouls are low, and defensive third touches are high (35+), portraying intelligent positioning and anticipation. They blend physical, technical, and cognitive traits seamlessly.\n\nTactical Fit: Built for elite positional play systems—whether it’s Guardiola’s City, Arteta’s Arsenal, or Inzaghi’s Inter. They are often the cornerstone of their team’s first phase and a key tool for control, capable of anchoring both high and mid blocks with equal assurance."
}


# PCA setup
X = df[center_back_features]
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
st.title("🎯 Center Backs")

page = st.sidebar.radio("Navigate", [
    "📌 Similarity Search", "🆚 Compare Players", "🧬 Cluster Profiles", "📊 Full Player Table"
])

if page == "📌 Similarity Search":
    player = st.selectbox("Select Player", df["Player"].unique())
    n = st.slider("Top N", 3, 30, 10)

    sim_list = compute_similarity(player, df_pca, cluster_labels, player_index)
    sim_list = [(player, 100.0)] + sim_list  # Ajoute le joueur source en haut
    df_sim = pd.DataFrame(sim_list, columns=["Player", "Similarity"])


    top_players_df = df_sim.head(n)
    top_players = top_players_df["Player"].tolist()

    # Merge similarity scores into main DataFrame
    detailed_df = df[df["Player"].isin(top_players)].copy()
    detailed_df = detailed_df.merge(top_players_df, on="Player")
    detailed_df = detailed_df.sort_values("Similarity", ascending=False)

    st.subheader("🧠 Similar Players with Full Stats")
    st.dataframe(detailed_df[["Player", "Similarity"] + meta_cols[1:] + ["Cluster Name"] + center_back_features].set_index("Player"))

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
    st.dataframe(df_filtered[meta_cols + ["Cluster Name"] + center_back_features].set_index("Player"))
