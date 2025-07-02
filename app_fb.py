import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("final_df_fb_with_clusters.csv")

meta_cols = ["Player", "Birthdate", "Age", "League", "Club", "Footed", "Nationality", "Position", "Rating", "Potential", "Minutes"]

# === FEATURE SET ===
fullbacks_features = [
    'Assists', 'Crosses', 'Crosses into Penalty Area',     # Chance creation from wide
    'Progressive Carries', 'Progressive Carrying Distance',# Advancing play on the flanks
    'Carries into Final Third',                            # Offensive contribution
    'Touches (Att 3rd)',                                   # Time spent high up the pitch
    'Pass Completion %', 'Pass Completion % (Long)',       # Passing security under pressure
    'Live-ball Passes', 'Passes Attempted (Long)',         # Ability to vary distribution
    'Progressive Passes', 'Progressive Passing Distance',  # Pushing team upfield
    'Switches',                                            # Switching play across field
    'Through Balls',                                       # Penetrative passes behind defense
    'Blocks', 'Shots Blocked',                             # Defensive contributions
    'Tackles (Def 3rd)', 'Tackles (Att 3rd)',              # Defensive actions at both ends
    'Dribbles Challenged', 'SCA (Live-ball Pass)',         # Defensive duels and shot creation
    'Fouls Committed', 'Fouls Drawn',                      # Defensive discipline and winning fouls
    'Aerials Won',                                         # Aerial presence (defensive/offensive)
    'Errors',                                              # Costly mistakes
    'Ball Recoveries'                                      # Winning back possession
]

radar_features =  [                              
    'npxG + xAG', 'Assists', 
    'Crosses',                                              # Chance creation from wide
    'Progressive Carrying Distance',                        # Advancing play on the flanks
    'Carries into Final Third',                             # Offensive contribution
    'Touches (Att 3rd)',                                    # Time spent high up the pitch
    'Pass Completion %',                                    # Passing security under pressure
    'Tackles (Def 3rd)',                                    # Defensive actions at both ends
    'Dribbles Challenged', 'SCA (Live-ball Pass)',          # Defensive duels and shot creation
    'Ball Recoveries'                                       # Winning back possession
]

cluster_names = {0: ('Inverted Facilitators', 102),
 1: ('Dynamic Engines', 101),
 2: ('Offensive Catalysts', 57),
 3: ('Robust Wide Stoppers', 127),
 4: ('Two-Way Modernisers', 91)}

# Cluster descriptions
cluster_descriptions = {
    "Inverted Facilitators": "This cluster aggregates players like Joško Gvardiol, Jules Koundé, Jurriën Timber, and Ben White—defenders traditionally raised as centre-backs but now operating as wide players, particularly in asymmetric back threes or fullback roles with conservative offensive mandates. Statistically, this group features the lowest involvement in progressive actions in the final third, as evidenced by modest averages in crosses into the penalty area (0.33), attacking third touches (17.18), and assists (0.06). However, they shine in high pass completion (82.1%) and defensive duels (1.26 tackles in the defensive third and 2.23 dribbles challenged), suggesting their primary function is ball retention, spatial compactness, and buildup stability.\n\nThese are not expansive fullbacks but rather secure facilitators who invert into midfield or back three structures to optimize circulation and balance. They are typically deployed in possession-dominant systems where the fullback is required to tuck inside—either to support the pivot (as in Guardiola’s 3-2-2-3) or reinforce rest defense principles.\n\nTactical Fit: Ideal for sides that play with fullback inversion principles—Manchester City, Arsenal, or Barcelona’s positional structures. Systems demanding high technical execution, numerical superiority in midfield, and proactive pressing value this profile highly.",

    "Dynamic Engines": "Players like Alejandro Balde, Federico Dimarco, and Antonee Robinson headline this cluster, which displays high values in transitional activity and direct width exploitation. With over 4.35 crosses per 90 and almost two carries into the final third (1.93), this group operates with a pronounced verticality. Although their pass completion is lowest among all clusters (76.0%), this is a reflection of their high-risk, high-reward style—often required to serve as the primary source of width and progression on the flanks.\n\nThey maintain significant carrying output (2.9 progressive carries; 98m distance) and also show sharp attacking output via live-ball shot-creating actions (2.0 SCA), indicating their role as the attacking outlet in wide zones. Their defensive output is moderate, suggesting structural support behind them or systems that allow aggressive positioning high up the pitch.\n\nTactical Fit: Suited for high-octane systems leveraging wide overlaps and quick transitions. Think of Atalanta, Inter under Inzaghi, or Premier League sides playing transitional 4-3-3 shapes. Also well-suited to wingback roles in 3-4-2-1s or 3-5-2s where the player is expected to own the entire flank.",

    "Offensive Catalysts": "Trent Alexander-Arnold, Achraf Hakimi, and Nuno Mendes define this cluster—a profile driven by elite offensive output and final-third orchestration. This group boasts the highest attacking third touches (24.3), most crosses into the penalty area (0.66), and the strongest passing creativity with 2.49 live-ball SCA. Their carrying distances (~120m) and touches in advanced zones point to their role as deep-lying chance creators or auxiliary playmakers from wide areas.\n\nInterestingly, they maintain decent balance in defensive metrics, with 1.06 defensive third tackles and 2.14 dribbles challenged—suggesting that while their primary value lies in progression and creation, they are not exempt from transitional recovery tasks. These are the fullbacks that transform wide spaces into launchpads for attack—often acting as a team’s second or third playmaker.\n\nTactical Fit: Best deployed in dominant teams with full-pitch occupation and possession control. Fits include Liverpool’s high-possession 4-3-3, PSG’s hybrid pressing system, or Bayern’s inverted transition schemes. Particularly effective in structures allowing for wide overloads and deep crossing profiles.",

    "Robust Wide Stoppers": "Represented by players such as Denzel Dumfries, James Justin, and Vitaliy Mykolenko, this group thrives in high-duel, high-intensity environments. Their statistical profile is marked by modest ball progression (only 1.06 carries into the final third and 0.22 assists) but solid defensive output: 1.26 defensive third tackles and 2.22 dribbles challenged per match. They are more effective in ball-stopping than ball-carrying, with less emphasis on elaborate buildup or deep progression.\n\nTheir aerial win rate is also higher than average (above 1.1 per 90), suggesting a physical profile that contributes to second-ball recoveries and back-post defending. These fullbacks function best in medium blocks or reactive systems where solidity, recovery speed, and defensive aggression are prioritized over ball circulation.\n\nTactical Fit: Perfectly suited for Premier League mid-table sides or Serie A teams deploying deeper defensive lines. Tactically reliable in 4-4-2 mid blocks, 5-3-2 systems, or man-oriented pressing schemes that prioritize vertical compression and 1v1 defending.",

    "Two-Way Modernisers": "This final group balances attacking and defensive duties with above-average contributions across most metrics without specializing to the extremes of creativity or suppression. Players like Lucas Digne, Daniel Muñoz, and Tyrick Mitchell populate this cluster, showcasing solid output in carrying (2.25 carries; 94.9m), crossing (3.3), and final-third involvement (~19 touches). Their pass completion sits at a respectable 79.4%, reflecting their involvement in both early-phase buildup and final-third circulation.\n\nDefensively, they remain consistent contributors, with 1.05 defensive third tackles and 2.14 dribbles challenged, and they rarely commit errors or fouls. These are dependable all-phase fullbacks that managers can trust to maintain tactical width, support midfielders, and protect wide defensive zones simultaneously.\n\nTactical Fit: These profiles are tactically flexible and fit a range of systems—from a 4-2-3-1 with overlapping responsibility to a balanced 4-3-3 or even wingback roles in more conservative 5-4-1 structures. They provide solutions for teams with less squad depth or that require multi-phase reliability from their fullbacks."
}

# PCA setup
X = df[fullbacks_features]
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
st.title("🎯 Full Backs")

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
    st.dataframe(detailed_df[["Player", "Similarity"] + meta_cols[1:] + ["Cluster Name"] + ["Goals"] + fullbacks_features].set_index("Player"))

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
    st.dataframe(df_filtered[meta_cols + ["Cluster Name"] + ["Goals"] + fullbacks_features].set_index("Player"))
