import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("final_df_mid_with_clusters.csv")

meta_cols = ["Player", "Birthdate", "Age", "League", "Club", "Footed", "Nationality", "Position", "Rating", "Potential", "Minutes"]

# === FEATURE SET ===
midfielders_features = [
    'Assists', 'npxG + xAG',                          # Playmaking and shot creation
    'Progressive Carries', 'Progressive Carrying Distance', # Ball progression through midfield
    'Touches (Att 3rd)', 'Touches (Def 3rd)',         # Influence in both halves
    'Pass Completion %', 'Live-ball Passes',          # Ball security
    'Progressive Passes', 'Progressive Passing Distance', # Advancing team forward
    'Passes Attempted (Long)', 'Pass Completion % (Long)',# Range of passing
    'Passes into Final Third', 'Switches',            # Penetrative and expansive passing
    'Through Balls',                                  # Breaking lines
    'SCA (Live-ball Pass)', 'SCA (Take-On)',          # Shot creation
    'SCA (Defensive Action)',                         # Disrupting opponents before shots
    'GCA (Live-ball Pass)', 'GCA (Defensive Action)', # Goal creation and last-ditch defending
    'Tackles (Mid 3rd)',                              # Defensive work rate in midfield
    'Tkl+Int',                                        # Ball-winning combined metric
    'Blocks',                                         # Blocking passes/shots
    'Ball Recoveries',                                # Regaining possession
    'Fouls Committed', 'Fouls Drawn',                 # Physical/technical battle
    'Aerials Won'                                     # Midfield duels                                     
]

radar_features = [
    'Goals', 'Assists', 'npxG + xAG',                          # Playmaking and shot creation
    'Progressive Carries',                            # Ball progression through midfield
    'Touches (Att 3rd)',                              # Influence in both halves
    'Pass Completion %',                              # Ball security
    'Progressive Passes',                             # Advancing team forward
    'SCA (Live-ball Pass)',
    'Tkl+Int',                                        # Ball-winning combined metric
    'Ball Recoveries',                                # Regaining possession
    'Fouls Committed'             # Physical/technical battle
]

cluster_names = {0: ('Advanced Playmakers', 86),
 1: ('Tempo Dictators', 36),
 2: ('Vertical Connectors', 114),
 3: ('Supportive Engines', 48),
 4: ('All-Round Carriers', 115),
 5: ('Ball-Winning Specialists', 97),
 6: ('Hybrid Workhorses', 117)}

# Cluster descriptions
cluster_descriptions = {
    "Advanced Playmakers": "This cluster groups players who combine creativity, progression, and goal involvement from advanced midfield positions. Profiles like Jude Bellingham, Martin Ødegaard, James Maddison, İlkay Gündoğan, and Bruno Guimarães define the archetype. These midfielders register the highest average non-penalty xG + xAG (0.33) and assists (0.18) across all clusters. They thrive in pockets between midfield and defense, operating as high-impact operators in the final third with progressive carries (2.56) and substantial carrying distance (~91m), while also sustaining over 21 touches in the attacking third per match.\n\nHowever, their defensive involvement remains moderate with only 2.18 tackles + interceptions, revealing a role structurally protected by deeper midfielders. These are your possession catalysts, dictating attacking rhythm through sharp positioning, tight-space awareness, and a vertical passing instinct (live-ball SCA: 2.34).\n\nTactical Fit: This profile flourishes in systems demanding final-third craft and structured rotations, notably in a 4-3-3 with false nine dynamics or a 3-2-5 with interior overloads. Think of them as your David Silva/Gündoğan types—fundamental to orchestrated final-third progression through combination play.",

    "Tempo Dictators": "This cluster is composed of elite all-phase midfielders like Luka Modrić, Frenkie de Jong, Pedri, Nicolò Barella, and Joshua Kimmich—architects who influence possession, tempo, and transitions. Statistically, they are the most complete: boasting the highest pass completion (86.2%), progressive passes (9.19), and live-ball passes (75.8). They carry the ball over long distances (~133m per 90, highest in dataset), indicating a blend of press resistance and dynamism.\n\nDefensively, they contribute robustly with 2.59 tackles + interceptions, high recoveries (5.63), and above-average presence in both thirds. Their value lies in sustaining control through the middle third while enabling verticality under pressure. They have strong shot-creating contributions (2.64) despite lower direct assist or scoring figures, underlining their indirect creative influence.\n\nTactical Fit: These midfielders are essential in possession-oriented structures (4-3-3, 3-2-4-1) demanding technical leadership and progression from deeper zones. They act as relay hubs, often the metronomes behind more creative or explosive partners. Think Busquets, Kroos, or Xavi in modernized versions.",

    "Vertical Connectors": "Midfielders in this cluster bridge defense and attack through purposeful movement and quick circulation, exemplified by players like Valverde, Gravenberch, Thomas Partey, and Zieliński. While their final-third output is modest (0.18 xG+xAG and 0.10 assists), they provide steady progression (1.31 carries, 6.4 progressive passes) and engage effectively in the defensive phase (2.93 tackles + interceptions, among highest).\n\nInterestingly, these profiles also register high touches in both defensive and attacking thirds, suggesting a box-to-box presence. They’re not primary creators, but rather glue players—cleaning transitions, sustaining width, and enabling stars around them to thrive.\n\nTactical Fit: Best suited for hybrid roles in double pivots or shuttling 8s in systems like a 4-2-3-1 or 3-1-4-2. Their athletic and technical balance allows them to perform high-tempo roles with defensive reliability and supporting structure.",

    "Supportive Engines": "Players like Moisés Caicedo, Alexis Mac Allister, and Conor Gallagher belong here—midfielders with high defensive output and moderate progression metrics, tasked with supporting transitions and covering tactical imbalances. They exhibit 1.13 progressive carries and 3.4 progressive passes, slightly below average in terms of direct attacking involvement.\n\nWhat sets them apart is their defensive robustness: 4.05 tackles + interceptions, 5.48 recoveries, and excellent block numbers. These are players who thrive when tasked with defensive responsibilities and structural control, acting as stabilizers within dynamic or aggressive systems.\n\nTactical Fit: Ideal in high-pressing or mid-block setups, particularly in roles requiring intense coverage (4-2-3-1 destroyer, 3-4-3 wing-shadows). Their function is as the lungs and legs—not headline makers, but indispensable for controlling transitions and plugging gaps.",

    "All-Round Carriers": "Featuring players like Mikel Merino, Frattesi, and McTominay, this cluster excels at ball-winning and transitional threat. Though their xG+xAG (0.21) is moderate, they contribute across the pitch with high carrying (1.08), solid physical output (4.14 recoveries, 2.77 tackles + interceptions), and the highest aerials won (1.28)—indicating duel strength and presence.\n\nThese midfielders are aggressive in their carries, tackle well in the mid-third, and are useful in both offensive and defensive transitions. They rank highest for fouls committed (1.50), which aligns with the profile of physically assertive midfielders disrupting rhythm and winning territory.\n\nTactical Fit: Perfect for systems needing high-volume carriers and box-to-box runners. Fits include 3-5-2 or 4-3-3 pressing shapes where second balls and duels dictate control. These are players who grind, carry, and compete—less refined, but tactically essential.",

    "Ball-Winning Specialists": "This group features Eduardo Camavinga, Wilfred Ndidi, Idrissa Gueye, and Florentino Luís—pure defensive specialists. They produce the highest defensive actions across the board, with 4.05 tackles + interceptions, 5.48 recoveries, and 1.68 fouls committed. Their attacking impact is minimal (0.19 xG+xAG), reflecting a role centered on screening, disrupting, and recycling.\n\nWhile their progression is modest (3.4 progressive passes, low final-third touches), they anchor midfield structures with elite ball-winning and positional discipline. They aren’t tasked with risk—rather, they clean the platform for others to shine.\n\nTactical Fit: Best used as single pivots in elite systems (e.g., 4-3-3 with high fullbacks) or double pivots for defensive coverage (4-2-3-1). These are modern-day Makeleles: low flair, high impact. The system breathes because they hold its spine.",

    "Hybrid Workhorses": "The final cluster consists of players like Zambo Anguissa, Kamara, and Robert Andrich—all-around profiles with balanced output across defense and progression. Statistically, they fall in the median of most metrics: 2.77 tackles + interceptions, 4.14 recoveries, 3.4 progressive passes, and ~75% pass completion.\n\nThese players are versatile and system-agnostic. While they don’t lead in any one category, they score solidly across all. Think of them as tactical Swiss knives—adaptable, reliable, but not elite specialists. This explains why they appear across a range of teams in rotational or stabilizing roles.\n\nTactical Fit: Valuable in squads needing tactical flexibility. Can slot into various systems (4-4-2 diamond, 3-4-3 hybrid press, or 4-3-3 box) depending on context. They offer coaching staff the ability to adjust shape and intensity without sacrificing structural coherence."
}


# PCA setup
X = df[midfielders_features]
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
st.title("🎯 Midfielders")

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
    st.dataframe(detailed_df[["Player", "Similarity"] + meta_cols[1:] + ["Cluster Name"] + ["Goals"] + midfielders_features].set_index("Player"))

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
    st.dataframe(df_filtered[meta_cols + ["Cluster Name"] + ["Goals"] + midfielders_features].set_index("Player"))
