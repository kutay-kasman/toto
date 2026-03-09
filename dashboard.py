"""
Streamlit web dashboard for football match predictions.
Shows predictions, historical accuracy, and key statistics.

This is a viewer-only version that reads directly from the database.
No Selenium or ML model dependencies required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Football Match Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def get_db_connection():
    """
    Get direct SQLite database connection (thread-safe).
    Creates a new connection each time to avoid threading issues in Streamlit Cloud.
    """
    db_path = Path("data/matches.db")
    if not db_path.exists():
        # Try alternative path
        db_path = Path("matches.db")
    
    if not db_path.exists():
        return None
    
    # Use check_same_thread=False for Streamlit Cloud compatibility
    # This allows the connection to be used across different threads
    return sqlite3.connect(str(db_path), check_same_thread=False)


@st.cache_data
def get_historical_matches():
    """Get historical matches from database."""
    db_path = Path("data/matches.db")
    if not db_path.exists():
        db_path = Path("matches.db")
    
    if not db_path.exists():
        return pd.DataFrame()
    
    # Create connection inside the cached function to avoid threading issues
    # Use context manager to ensure proper cleanup
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as conn:
            df = pd.read_sql_query("""
                SELECT id,
                       match_date,
                       home_team as 'Ev Sahibi Takım',
                       away_team as 'Deplasman Takımı',
                       result as 'Mac Sonucu',
                       home_goals as 'Home_Goals',
                       away_goals as 'Away_Goals'
                FROM historical_matches
                ORDER BY match_date, id
            """, conn)
            # Return a copy to ensure no connection references remain
            return df.copy()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data
def get_upcoming_matches():
    """Get upcoming matches from database."""
    db_path = Path("data/matches.db")
    if not db_path.exists():
        db_path = Path("matches.db")
    
    if not db_path.exists():
        return pd.DataFrame()
    
    # Create connection inside the cached function to avoid threading issues
    # Use context manager to ensure proper cleanup
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as conn:
            df = pd.read_sql_query("""
                SELECT home_team as 'Ev Sahibi Takım',
                       away_team as 'Deplasman Takımı',
                       match_date
                FROM upcoming_matches
                ORDER BY match_date, id
            """, conn)
            # Return a copy to ensure no connection references remain
            return df.copy()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data
def get_predictions(limit=1000):
    """Get predictions from database."""
    db_path = Path("data/matches.db")
    if not db_path.exists():
        db_path = Path("matches.db")
    
    if not db_path.exists():
        return pd.DataFrame()
    
    # Create connection inside the cached function to avoid threading issues
    # Use context manager to ensure proper cleanup
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as conn:
            df = pd.read_sql_query("""
                SELECT home_team, away_team, predicted_result,
                       probability_1, probability_x, probability_2,
                       actual_result, is_correct, prediction_date
                FROM predictions
                ORDER BY prediction_date DESC
                LIMIT ?
            """, conn, params=(limit,))
            # Return a copy to ensure no connection references remain
            return df.copy()
    except Exception as e:
        return pd.DataFrame()


@st.cache_data
def get_prediction_accuracy():
    """Calculate overall prediction accuracy."""
    db_path = Path("data/matches.db")
    if not db_path.exists():
        db_path = Path("matches.db")
    
    if not db_path.exists():
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}
    
    # Create connection inside the cached function to avoid threading issues
    # Use context manager to ensure proper cleanup
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(is_correct) as correct,
                    AVG(is_correct) as accuracy
                FROM predictions
                WHERE actual_result IS NOT NULL
            """)
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                return {
                    'total': result[0],
                    'correct': result[1] or 0,
                    'accuracy': (result[2] or 0) * 100
                }
            return {'total': 0, 'correct': 0, 'accuracy': 0.0}
    except Exception as e:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}


def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">⚽ Football Match Prediction Dashboard by Kutay</h1>', 
                unsafe_allow_html=True)
    
    # Check if database exists
    db_path = Path("data/matches.db")
    if not db_path.exists():
        db_path = Path("matches.db")
    
    if not db_path.exists():
        st.error("⚠️ Database not found!")
        st.info("Please ensure `data/matches.db` exists. This dashboard reads from the database only.")
        return
    
    # Sidebar
    page = st.sidebar.radio(
        "Select Page",
        ["This Week's Predictions", "Combination Performance", "Historical Accuracy", "Deep Analysis", "Model Statistics", "Data Overview"]
    )
    
    if page == "This Week's Predictions":
        show_predictions()
    elif page == "Combination Performance":
        show_combination_performance()
    elif page == "Historical Accuracy":
        show_accuracy()
    elif page == "Model Statistics":
        show_model_stats()
    elif page == "Data Overview":
        show_data_overview()
    elif page == "Deep Analysis":
        show_deep_analysis()


def show_deep_analysis():
    """Display deep statistics and analysis."""
    st.header("🔬 Deep Analysis")
    
    # Get all data
    historical = get_historical_matches()
    
    if historical.empty:
        st.warning("No historical data available.")
        return
        
    # Tabs for different analysis modes
    tab1, tab2 = st.tabs(["🏆 Team Analysis", "⚔️ Head-to-Head"])
    
    # --- Tab 1: Team Analysis ---
    with tab1:
        st.subheader("Team Performance Analysis")
        
        # Team Selector
        all_teams = sorted(list(set(historical['Ev Sahibi Takım'].tolist() + 
                                  historical['Deplasman Takımı'].tolist())))
        selected_team = st.selectbox("Select Team", all_teams)
        
        if selected_team:
            # Filter matches involving this team
            team_matches = historical[
                (historical['Ev Sahibi Takım'] == selected_team) | 
                (historical['Deplasman Takımı'] == selected_team)
            ].sort_values('id')  # Assuming 'id' correlates with time
            
            # --- Key Stats ---
            total_games = len(team_matches)
            wins = 0
            draws = 0
            losses = 0
            goals_scored = 0
            goals_conceded = 0
            
            for _, match in team_matches.iterrows():
                is_home = match['Ev Sahibi Takım'] == selected_team
                result = match['Mac Sonucu']
                
                # W/D/L
                if result == 'X':
                    draws += 1
                elif (is_home and result == '1') or (not is_home and result == '2'):
                    wins += 1
                else:
                    losses += 1
                    
                # Goals (handling potential None values)
                h_goals = match.get('Home_Goals')
                a_goals = match.get('Away_Goals')
                
                if pd.notna(h_goals) and pd.notna(a_goals):
                    if is_home:
                        goals_scored += int(h_goals)
                        goals_conceded += int(a_goals)
                    else:
                        goals_scored += int(a_goals)
                        goals_conceded += int(h_goals)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Games", total_games)
            with col2:
                win_rate = (wins/total_games*100) if total_games > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                avg_scored = (goals_scored/total_games) if total_games > 0 else 0
                st.metric("Avg Goals Scored", f"{avg_scored:.1f}")
            with col4:
                avg_conceded = (goals_conceded/total_games) if total_games > 0 else 0
                st.metric("Avg Goals Conceded", f"{avg_conceded:.1f}")
            
            # --- Visuals ---
            
            # 1. Result Distribution Pie Chart
            fig_pie = px.pie(
                values=[wins, draws, losses],
                names=['Win', 'Draw', 'Loss'],
                title=f"{selected_team} Match Outcomes",
                color_discrete_map={'Win': '#2ca02c', 'Draw': '#ff7f0e', 'Loss': '#d62728'}
            )
            st.plotly_chart(fig_pie, width='stretch')
            
            # 2. Form Guide (Last 10 Matches)
            st.subheader("Recent Form (Last 10 Matches)")
            last_10 = team_matches.tail(10).copy()
            
            form_data = []
            for _, match in last_10.iterrows():
                is_home = match['Ev Sahibi Takım'] == selected_team
                opponent = match['Deplasman Takımı'] if is_home else match['Ev Sahibi Takım']
                result = match['Mac Sonucu']
                
                outcome = 'Draw'
                color = 'gray'
                if (is_home and result == '1') or (not is_home and result == '2'):
                    outcome = 'Win'
                    color = 'green'
                elif (is_home and result == '2') or (not is_home and result == '1'):
                    outcome = 'Loss'
                    color = 'red'
                
                score = "N/A"
                if pd.notna(match.get('Home_Goals')):
                    score = f"{int(match['Home_Goals'])}-{int(match['Away_Goals'])}"
                
                form_data.append({
                    'Opponent': opponent,
                    'Result': outcome,
                    'Score': score,
                    'Venue': 'Home' if is_home else 'Away',
                    'Color': color
                })
            
            form_df = pd.DataFrame(form_data)
            st.table(form_df)

    # --- Tab 2: Head-to-Head ---
    with tab2:
        st.subheader("Head-to-Head Comparison")
        
        col_a, col_b = st.columns(2)
        with col_a:
            team_a = st.selectbox("Team A", all_teams, index=0)
        with col_b:
            team_b = st.selectbox("Team B", all_teams, index=1 if len(all_teams) > 1 else 0)
            
        if team_a != team_b:
            # Find H2H matches
            h2h_matches = historical[
                ((historical['Ev Sahibi Takım'] == team_a) & (historical['Deplasman Takımı'] == team_b)) |
                ((historical['Ev Sahibi Takım'] == team_b) & (historical['Deplasman Takımı'] == team_a))
            ]
            
            if h2h_matches.empty:
                st.info("No historical matches found between these teams.")
            else:
                st.write(f"Found {len(h2h_matches)} matches.")
                
                # Calculate stats
                a_wins = 0
                b_wins = 0
                draws = 0
                
                for _, match in h2h_matches.iterrows():
                    res = match['Mac Sonucu']
                    if res == 'X':
                        draws += 1
                    elif match['Ev Sahibi Takım'] == team_a:
                        if res == '1': a_wins += 1
                        else: b_wins += 1
                    else: # match['Ev Sahibi Takım'] == team_b
                        if res == '1': b_wins += 1
                        else: a_wins += 1
                
                # Display Bars
                st.write(f"**{team_a} Wins**: {a_wins} | **Draws**: {draws} | **{team_b} Wins**: {b_wins}")
                
                fig_h2h = go.Figure(data=[
                    go.Bar(name=team_a, x=['Wins'], y=[a_wins], marker_color='#1f77b4'),
                    go.Bar(name='Draw', x=['Wins'], y=[draws], marker_color='#7f7f7f'),
                    go.Bar(name=team_b, x=['Wins'], y=[b_wins], marker_color='#ff7f0e')
                ])
                fig_h2h.update_layout(barmode='stack', title="Head-to-Head Results")
                st.plotly_chart(fig_h2h, width='stretch')
                
                # Match History Table
                st.subheader("Match History")
                display_cols = ['Ev Sahibi Takım', 'Deplasman Takımı', 'Mac Sonucu', 'Home_Goals', 'Away_Goals']
                st.dataframe(h2h_matches[display_cols].sort_index(ascending=False), hide_index=True)
                
        else:
            st.warning("Please select distinct teams.")


def calculate_entropy(p1, px, p2):
    """
    Calculate Shannon entropy for a match prediction.
    
    Higher entropy = more uncertain match
    Range: [0, ~1.58] (max with 3 equally likely outcomes)
    
    Args:
        p1: Probability of home win
        px: Probability of draw  
        p2: Probability of away win
    
    Returns:
        Entropy value (float)
    """
    probs = [p for p in [p1, px, p2] if p > 0]  # Remove zeros
    
    if not probs:
        return 0.0
    
    # Shannon entropy: H(X) = -Σ p(x) * log2(p(x))
    entropy = -sum(p * np.log2(p) for p in probs)
    return entropy


def generate_prediction_combinations(predictions_df, n_combinations=7, top_k=5):
    """
    Generate smart prediction combinations based on entropy (uncertainty).
    
    Strategy:
    1. Calculate entropy for each match
    2. Identify top-K most uncertain matches
    3. Fix certain matches to their best prediction
    4. Generate combinations only varying uncertain matches
    
    This dramatically reduces Monte Carlo attempts needed!
    
    Args:
        predictions_df: DataFrame with predictions and probabilities
        n_combinations: Number of combinations to generate (default: 7)
        top_k: Number of most uncertain matches to vary (default: 5)
    
    Returns:
        List of dicts with 'predictions' (list), 'score' (float), and metadata
    """
    if predictions_df.empty:
        return []
    
    # Extract match info with entropy
    matches = []
    entropies = []
    
    for _, row in predictions_df.iterrows():
        p1 = row['probability_1']
        px = row['probability_x']
        p2 = row['probability_2']
        
        # Calculate entropy for this match
        entropy = calculate_entropy(p1, px, p2)
        entropies.append(entropy)
        
        probs = {'1': p1, 'X': px, '2': p2}
        sorted_outcomes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        matches.append({
            'home': row['home_team'],
            'away': row['away_team'],
            'probs': probs,
            'sorted': sorted_outcomes,
            'entropy': entropy,
            'best': sorted_outcomes[0][0]
        })
    
    # Find top-K most uncertain matches (highest entropy)
    uncertain_indices = sorted(
        range(len(matches)),
        key=lambda i: matches[i]['entropy'],
        reverse=True
    )[:min(top_k, len(matches))]
    
    logger.info(f"Smart combination generation: {len(matches)} matches, {len(uncertain_indices)} uncertain")
    logger.info(f"Uncertain match indices: {uncertain_indices}")
    
    # Build base prediction (certain matches use best prediction)
    base_predictions = []
    for idx, match in enumerate(matches):
        if idx in uncertain_indices:
            base_predictions.append(None)  # Will vary
        else:
            base_predictions.append(match['best'])  # Fixed to best
    
    combinations = []
    
    # Combination 1: Pure greedy (all best predictions)
    greedy_pred = [m['best'] for m in matches]
    greedy_score = np.prod([m['probs'][m['best']] for m in matches])
    combinations.append({
        'predictions': greedy_pred,
        'score': greedy_score,
        'strategy': 'greedy'
    })
    
    # Generate combinations by varying only uncertain matches
    from itertools import product
    
    # Get all possible outcomes for uncertain matches
    uncertain_options = []
    for idx in uncertain_indices:
        # For each uncertain match, try top 2 or 3 best options
        match = matches[idx]
        # Use outcomes with probability > 15% (avoid very unlikely ones)
        viable = [opt[0] for opt in match['sorted'] if opt[1] > 0.15]
        if len(viable) < 2:
            viable = [opt[0] for opt in match['sorted'][:2]]  # At least top 2
        uncertain_options.append(viable)
    
    # Generate all permutations for uncertain matches
    seen = {tuple(greedy_pred)}
    
    for perm in product(*uncertain_options):
        if len(combinations) >= n_combinations:
            break
        
        # Build full prediction
        full_prediction = base_predictions.copy()
        for i, idx in enumerate(uncertain_indices):
            full_prediction[idx] = perm[i]
        
        # Check uniqueness
        pred_tuple = tuple(full_prediction)
        if pred_tuple in seen:
            continue
        seen.add(pred_tuple)
        
        # Calculate score
        score = np.prod([matches[i]['probs'][p] for i, p in enumerate(full_prediction)])
        
        combinations.append({
            'predictions': full_prediction,
            'score': score,
            'strategy': 'smart_topk'
        })
    
    # If we don't have enough combinations, add a few random ones
    attempt = 0
    while len(combinations) < n_combinations and attempt < 100:
        attempt += 1
        
        variant = base_predictions.copy()
        for idx in uncertain_indices:
            # Random choice from viable options
            options = [opt[0] for opt in matches[idx]['sorted'][:2]]
            variant[idx] = np.random.choice(options)
        
        pred_tuple = tuple(variant)
        if pred_tuple not in seen:
            seen.add(pred_tuple)
            score = np.prod([matches[i]['probs'][p] for i, p in enumerate(variant)])
            combinations.append({
                'predictions': variant,
                'score': score,
                'strategy': 'random_fill'
            })
    
    # Sort by score (highest confidence first)
    combinations.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info(f"Generated {len(combinations)} combinations (target: {n_combinations})")
    
    return combinations[:n_combinations]


def show_prediction_combinations(predictions_df):
    """Display alternative prediction combinations."""
    st.subheader("🎯 Alternative Prediction Combinations")
    st.info("Sistemimiz, olasılıklara göre 5-7 farklı tahmin kombinasyonu oluşturur. "
            "En yüksek güven skoruna sahip kombinasyon en üstte görünür.")
    
    # Generate combinations
    combinations = generate_prediction_combinations(predictions_df, n_combinations=7)
    
    if not combinations:
        st.warning("Kombinasyon oluşturulamadı.")
        return
    
    # Create display table
    combo_data = []
    for i, combo in enumerate(combinations, 1):
        pred_str = " ".join(combo['predictions'])
        score_pct = combo['score'] * 100
        
        combo_data.append({
            '#': i,
            'Kombinasyon': pred_str,
            'Güven Skoru': f"{score_pct:.2f}%",
            'Score_Raw': combo['score']  # For highlighting
        })
    
    combo_df = pd.DataFrame(combo_data)
    
    # Highlight top 3
    def highlight_top(row):
        if row['#'] == 1:
            return ['background-color: #90EE90'] * len(row)  # Light green
        elif row['#'] == 2:
            return ['background-color: #FFD700'] * len(row)  # Gold
        elif row['#'] == 3:
            return ['background-color: #FFA500'] * len(row)  # Orange
        return [''] * len(row)
    
    # Display styled table
    display_df = combo_df[['#', 'Kombinasyon', 'Güven Skoru']]
    st.dataframe(
        display_df.style.apply(highlight_top, axis=1),
        hide_index=True,
        use_container_width=True
    )
    
    # Expandable section for detailed view
    with st.expander("📊 Detaylı Kombinasyon Karşılaştırması"):
        # Match-by-match comparison
        match_names = [f"{row['home_team']} vs {row['away_team']}" 
                      for _, row in predictions_df.iterrows()]
        
        comparison_data = []
        for i, combo in enumerate(combinations[:5], 1):  # Top 5
            row_data = {'Kombinasyon': f"#{i}"}
            for j, pred in enumerate(combo['predictions']):
                row_data[f"Maç {j+1}"] = pred
            row_data['Skor'] = f"{combo['score']*100:.2f}%"
            comparison_data.append(row_data)
        
        st.table(pd.DataFrame(comparison_data))


def show_predictions():
    """Display this week's predictions."""
    st.header("📊 This Week's Predictions")
    
    # Get upcoming matches and predictions
    upcoming = get_upcoming_matches()
    predictions = get_predictions(limit=15)
    
    if upcoming.empty:
        st.warning("No upcoming matches found. Please run the prediction pipeline first.")
        st.info("Run: `python main.py --mode predict`")
        return
    
    # Merge upcoming matches with predictions
    if not predictions.empty:
        pred_dict = {
            (row['home_team'], row['away_team']): row 
            for _, row in predictions.iterrows()
        }
        
        results = []
        for _, match in upcoming.iterrows():
            home = match['Ev Sahibi Takım']
            away = match['Deplasman Takımı']
            key = (home, away)
            
            if key in pred_dict:
                pred = pred_dict[key]
                results.append({
                    'Match': f"{home} vs {away}",
                    'Prediction': pred['predicted_result'],
                    'Probability 1': f"{pred['probability_1']*100:.1f}%",
                    'Probability X': f"{pred['probability_x']*100:.1f}%",
                    'Probability 2': f"{pred['probability_2']*100:.1f}%",
                    'Confidence': max(pred['probability_1'], pred['probability_x'], pred['probability_2']) * 100
                })
        
        if results:
            df_results = pd.DataFrame(results)
            
            # Display predictions table
            st.dataframe(df_results, width='stretch', hide_index=True)
            
            # Coupon format
            st.subheader("🎫 Coupon Format")
            coupon = " ".join(df_results['Prediction'].tolist())
            st.code(coupon, language=None)
            
            # Confidence distribution
            st.subheader("📈 Prediction Confidence Distribution")
            fig = px.histogram(
                df_results, 
                x='Confidence',
                nbins=20,
                title="Distribution of Prediction Confidence",
                labels={'Confidence': 'Confidence (%)', 'count': 'Number of Matches'}
            )
            st.plotly_chart(fig, width='stretch')
            
            # Probability visualization
            st.subheader("🎲 Probability Heatmap")
            prob_data = []
            for _, row in df_results.iterrows():
                prob_data.append({
                    'Match': row['Match'],
                    '1 (Home)': float(row['Probability 1'].rstrip('%')),
                    'X (Draw)': float(row['Probability X'].rstrip('%')),
                    '2 (Away)': float(row['Probability 2'].rstrip('%'))
                })
            
            prob_df = pd.DataFrame(prob_data)
            prob_df = prob_df.set_index('Match')
            
            fig = px.imshow(
                prob_df.T,
                labels=dict(x="Match", y="Outcome", color="Probability (%)"),
                aspect="auto",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, width='stretch')
            
            # Add alternative combinations section
            st.markdown("---")
            show_prediction_combinations(predictions)
        else:
            st.warning("No predictions found for upcoming matches.")
    else:
        st.warning("No predictions available. Please run the prediction pipeline.")


def show_accuracy():
    """Display historical prediction accuracy."""
    st.header("📈 Historical Prediction Accuracy")
    
    predictions = get_predictions(limit=1000)
    
    if predictions.empty:
        st.info("No predictions with results yet. Check back after matches are played.")
        return
    
    # Filter predictions with actual results
    completed = predictions[predictions['actual_result'].notna()].copy()
    
    if completed.empty:
        st.info("No completed predictions yet.")
        return
    
    # Overall accuracy
    accuracy_data = get_prediction_accuracy()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", accuracy_data['total'])
    with col2:
        st.metric("Correct Predictions", accuracy_data['correct'])
    with col3:
        st.metric("Accuracy", f"{accuracy_data['accuracy']:.2f}%")
    with col4:
        incorrect = accuracy_data['total'] - accuracy_data['correct']
        st.metric("Incorrect", incorrect)
    
    # Accuracy over time
    completed['prediction_date'] = pd.to_datetime(completed['prediction_date'])
    completed = completed.sort_values('prediction_date')
    completed['cumulative_accuracy'] = completed['is_correct'].expanding().mean() * 100
    
    st.subheader("📊 Accuracy Over Time")
    fig = px.line(
        completed,
        x='prediction_date',
        y='cumulative_accuracy',
        title="Cumulative Prediction Accuracy",
        labels={'prediction_date': 'Date', 'cumulative_accuracy': 'Accuracy (%)'}
    )
    fig.add_hline(y=33.33, line_dash="dash", line_color="red", 
                  annotation_text="Random Guess (33.33%)")
    st.plotly_chart(fig, width='stretch')
    
    # Confusion matrix
    st.subheader("🔍 Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    
    y_true = completed['actual_result'].values
    y_pred = completed['predicted_result'].values
    
    cm = confusion_matrix(y_true, y_pred, labels=['1', 'X', '2'])
    cm_df = pd.DataFrame(
        cm,
        index=['Actual 1', 'Actual X', 'Actual 2'],
        columns=['Predicted 1', 'Predicted X', 'Predicted 2']
    )
    
    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, width='stretch')
    
    # Recent predictions table
    st.subheader("📋 Recent Predictions")
    display_cols = ['home_team', 'away_team', 'predicted_result', 
                   'actual_result', 'is_correct', 'prediction_date']
    st.dataframe(
        completed[display_cols].head(20),
        width='stretch',
        hide_index=True
    )


def show_model_stats():
    """Display model statistics and features."""
    st.header("🤖 Model Statistics")
    
    db_path = Path("data/matches.db")
    if not db_path.exists():
        db_path = Path("matches.db")
    
    if not db_path.exists():
        st.error("Database not found. Please ensure data/matches.db exists.")
        return
    
    # Create connection directly (not cached, so no threading issues)
    # Use context manager to ensure proper cleanup
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as conn:
            # Get model performance history from database
            perf_df = pd.read_sql_query("""
                SELECT model_name, accuracy, precision, recall, f1_score, training_date
                FROM model_performance
                ORDER BY training_date DESC
                LIMIT 10
            """, conn)
            
            if not perf_df.empty:
                st.subheader("Model Information")
                latest = perf_df.iloc[0]
                st.info(f"**Model Type:** {latest['model_name']}\n\n"
                       f"**Latest Accuracy:** {latest['accuracy']*100:.2f}%\n\n"
                       f"**Latest Training:** {latest['training_date']}")
                
                # Performance over time
                st.subheader("📊 Model Performance Over Time")
                perf_df['training_date'] = pd.to_datetime(perf_df['training_date'])
                fig = px.line(
                    perf_df,
                    x='training_date',
                    y='accuracy',
                    title="Model Accuracy Over Time",
                    labels={'accuracy': 'Accuracy', 'training_date': 'Training Date'}
                )
                st.plotly_chart(fig, width='stretch')
                
                st.plotly_chart(fig, width='stretch')
                
                # --- Optimized Hyperparameters ---
                try:
                    import joblib
                    model_path = Path("models/xgboost_model.pkl")
                    if model_path.exists():
                        model_data = joblib.load(model_path)
                        best_params = model_data.get('best_params', None)
                        if best_params:
                            st.subheader("⚙️ Optimized Hyperparameters")
                            st.json(best_params)
                        else:
                            st.info("Model was trained without optimization or params not saved.")
                except Exception as e:
                    st.warning(f"Could not load hyperparameters: {e}")

                # Latest metrics
                st.subheader("📈 Latest Model Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{latest['accuracy']*100:.2f}%")
                with col2:
                    st.metric("Precision", f"{latest['precision']*100:.2f}%")
                with col3:
                    st.metric("Recall", f"{latest['recall']*100:.2f}%")
                with col4:
                    st.metric("F1 Score", f"{latest['f1_score']*100:.2f}%")
            else:
                st.info("No model performance data available. Model metrics are logged during training.")
    except Exception as e:
        st.warning(f"Could not load model statistics: {e}")


def show_combination_performance():
    """Display combination tracking and performance page."""
    st.header("🎯 Combination Performance Tracker")
    
    # Import database
    from src.database import MatchDatabase
    from src.combination_evaluator import evaluate_combinations, format_result_summary
    
    db = MatchDatabase()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Current Week", "📈 Historical Performance", "✍️ Enter Results"])
    
    # --- Tab 1: Current Week Combinations ---
    with tab1:
        st.subheader("This Week's Combinations")
        
        latest_batch = db.get_latest_batch_id()
        
        if not latest_batch:
            st.info("No combinations generated yet. Run predictions first: `python main.py --mode predict`")
        else:
            st.info(f"**Batch ID:** {latest_batch}")
            
            combinations = db.get_combinations_for_batch(latest_batch)
            
            if combinations:
                # Display all combinations in a table
                combo_data = []
                for combo in combinations:
                    pred_str = " ".join(combo['predictions'])
                    score_pct = combo['score'] * 100
                    
                    combo_data.append({
                        'Rank': combo['rank'],
                        'Combination': pred_str,
                        'Confidence Score': f"{score_pct:.4f}%"
                    })
                
                combo_df = pd.DataFrame(combo_data)
                
                # Highlight top 3
                def highlight_top(row):
                    if row['Rank'] == 1:
                        return ['background-color: #90EE90'] * len(row)
                    elif row['Rank'] == 2:
                        return ['background-color: #FFD700'] * len(row)
                    elif row['Rank'] == 3:
                        return ['background-color: #FFA500'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(
                    combo_df.style.apply(highlight_top, axis=1),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Show if already evaluated
                stats_df = db.get_combination_statistics()
                if not stats_df.empty and latest_batch in stats_df['batch_id'].values:
                    result_row = stats_df[stats_df['batch_id'] == latest_batch].iloc[0]
                    
                    if pd.notna(result_row['winning_rank']):
                        st.success(f"✅ **Sonuç:** {int(result_row['winning_rank'])}. kombinasyon! "
                                 f"({int(result_row['total_correct'])}/{int(result_row['total_matches'])} doğru)")
                    else:
                        st.warning(f"⚠️ Hiçbir kombinasyon tam isabet değil. "
                                 f"En iyi: {int(result_row['total_correct'])}/{int(result_row['total_matches'])} doğru")
            else:
                st.warning("No combinations found for this batch.")
    
    # --- Tab 2: Historical Performance ---
    with tab2:
        st.subheader("Historical Performance Statistics")
        
        stats_df = db.get_combination_statistics()
        
        if stats_df.empty:
            st.info("No historical data yet. Enter results in the 'Enter Results' tab.")
        else:
            # Overall statistics
            total_batches = len(stats_df)
            perfect_matches = stats_df['winning_rank'].notna().sum()
            perfect_rate = (perfect_matches / total_batches * 100) if total_batches > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Weeks Evaluated", total_batches)
            with col2:
                st.metric("Perfect Matches", f"{perfect_matches}")
            with col3:
                st.metric("Perfect Match Rate", f"{perfect_rate:.1f}%")
            
            # Distribution of winning ranks
            st.subheader("📊 Which Combination Wins Most Often?")
            
            winning_ranks = stats_df[stats_df['winning_rank'].notna()]['winning_rank'].astype(int)
            
            if len(winning_ranks) > 0:
                rank_counts = winning_ranks.value_counts().sort_index()
                
                fig = px.bar(
                    x=rank_counts.index,
                    y=rank_counts.values,
                    labels={'x': 'Combination Rank', 'y': 'Number of Wins'},
                    title="Distribution of Winning Combinations"
                )
                fig.update_xaxis(tickmode='linear')
                st.plotly_chart(fig, use_container_width=True)
                
                # Pie chart
                fig_pie = px.pie(
                    values=rank_counts.values,
                    names=[f"#{rank}" for rank in rank_counts.index],
                    title="Winning Combination Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No perfect matches yet.")
            
            # Average accuracy per rank
            st.subheader("📈 Average Accuracy by Rank")
            avg_accuracy = stats_df.groupby('winning_rank')['accuracy'].mean().sort_index()
            
            if not avg_accuracy.empty:
                fig = px.line(
                    x=avg_accuracy.index,
                    y=avg_accuracy.values,
                    markers=True,
                    labels={'x': 'Winning Rank', 'y': 'Average Accuracy (%)'},
                    title="Average Accuracy for Each Winning Rank"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline
            st.subheader("📅 Performance Over Time")
            stats_df['evaluated_at'] = pd.to_datetime(stats_df['evaluated_at'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stats_df['evaluated_at'],
                y=stats_df['winning_rank'],
                mode='lines+markers',
                name='Winning Rank',
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Winning Combination Rank Over Time",
                xaxis_title="Date",
                yaxis_title="Winning Rank",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed History")
            display_df = stats_df[['batch_id', 'winning_rank', 'total_correct', 'total_matches', 'accuracy', 'evaluated_at']]
            display_df.columns = ['Batch', 'Winning Rank', 'Correct', 'Total', 'Accuracy (%)', 'Evaluated']
            st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # --- Tab 3: Enter Results ---
    with tab3:
        st.subheader("✍️ Enter Actual Results")
        
        # Get all available batches (from prediction_combinations table)
        try:
            all_batches_df = pd.read_sql_query(
                "SELECT DISTINCT batch_id FROM prediction_combinations ORDER BY batch_id DESC",
                db.get_db_connection() if hasattr(db, 'get_db_connection') else sqlite3.connect(db.db_path)
            )
            
            if all_batches_df.empty:
                st.warning("No batches available. Generate predictions first: `python main.py --mode predict`")
                return
            
            available_batches = all_batches_df['batch_id'].tolist()
            
            # Batch selector
            st.write("**Select Batch to Evaluate:**")
            selected_batch = st.selectbox(
                "Batch",
                options=available_batches,
                format_func=lambda x: f"{x} {'(Latest)' if x == available_batches[0] else ''}",
                label_visibility="collapsed"
            )
            
            # Session-state key scoped to the selected batch
            reeval_key = f"re_evaluate_{selected_batch}"
            if reeval_key not in st.session_state:
                st.session_state[reeval_key] = False
            
            # Check if already evaluated
            stats_df = db.get_combination_statistics()
            already_evaluated = (
                not stats_df.empty
                and selected_batch in stats_df['batch_id'].values
                and not st.session_state[reeval_key]
            )
            
            if already_evaluated:
                result_row = stats_df[stats_df['batch_id'] == selected_batch].iloc[0]
                st.info(f"ℹ️ Bu batch zaten değerlendirilmiş: "
                       f"{'Kazanan: #' + str(int(result_row['winning_rank'])) if pd.notna(result_row['winning_rank']) else 'Tam isabet yok'} "
                       f"({int(result_row['total_correct'])}/{int(result_row['total_matches'])} doğru)")
                
                # Show Monte Carlo results if available
                monte_carlo_df = db.get_monte_carlo_statistics()
                if not monte_carlo_df.empty and selected_batch in monte_carlo_df['batch_id'].values:
                    st.write("---")
                    st.subheader("🎲 Monte Carlo Simülasyonu Sonuçları")
                    
                    mc_row = monte_carlo_df[monte_carlo_df['batch_id'] == selected_batch].iloc[0]
                    
                    attempts   = int(mc_row['attempts_to_perfect'])
                    expected   = int(mc_row['expected_attempts'])
                    prob       = mc_row['theoretical_probability'] * 100
                    is_success = bool(mc_row.get('is_success', 0))
                    best_correct  = mc_row.get('best_correct')
                    best_attempt  = mc_row.get('best_attempt')
                    total_matches = mc_row.get('total_matches')
                    
                    if is_success:
                        st.success(f"✅ Tam isabet bulundu! {attempts:,}. denemede.")
                        if attempts < expected:
                            diff_pct = ((expected - attempts) / expected) * 100
                            st.info(f"✨ Şanslı! ({diff_pct:.0f}% daha az deneme)")
                        elif attempts > expected:
                            diff_pct = ((attempts - expected) / expected) * 100
                            st.info(f"😅 Şanssız ({diff_pct:.0f}% daha fazla deneme)")
                        else:
                            st.info("🎯 Tam beklenen kadar!")
                    else:
                        if pd.notna(best_correct) and pd.notna(total_matches):
                            st.warning(
                                f"⚠️ {attempts:,} denemede tam isabet bulunamadı. "
                                f"En iyi skor: **{int(best_correct)}/{int(total_matches)}** "
                                f"({int(best_attempt):,}. denemede)"
                            )
                        else:
                            st.warning(f"⚠️ {attempts:,} denemede tam isabet bulunamadı.")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Teorik Olasılık", f"{prob:.3f}%")
                    with col2:
                        st.metric("Beklenen Deneme", f"{expected:,}")
                    with col3:
                        label = "Tam İsabet Denemesi" if is_success else "Toplam Deneme"
                        st.metric(label, f"{attempts:,}")
                
                st.write("")
                if st.button("🔄 Yeniden Değerlendir"):
                    st.session_state[reeval_key] = True
                    st.rerun()
                return
            
            # Get combinations and predictions for selected batch
            combinations = db.get_combinations_for_batch(selected_batch)
            predictions = db.get_predictions_by_batch(selected_batch)
            
            if combinations and not predictions.empty:
                # Show matches
                st.write(f"**Maçlar ({len(predictions)} adet):**")
                
                # Create a form for entering results
                with st.form("result_entry_form"):
                    actual_results = []
                    
                    for idx, (_, row) in enumerate(predictions.iterrows()):
                        match_str = f"{row['home_team']} vs {row['away_team']}"
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{idx+1}.** {match_str}")
                        with col2:
                            result = st.selectbox(
                                "Result",
                                options=['1', 'X', '2'],
                                key=f"result_{idx}",
                                label_visibility="collapsed"
                            )
                            actual_results.append(result)
                    
                    submitted = st.form_submit_button("🎯 Evaluate Combinations")
                    
                    if submitted:
                        # Evaluate combinations
                        evaluation = evaluate_combinations(combinations, actual_results)
                        
                        # Save to database
                        db.save_combination_result(
                            batch_id=selected_batch,
                            winning_rank=evaluation['winning_rank'],
                            total_correct=evaluation['best_correct'],
                            total_matches=evaluation['total_matches']
                        )
                        
                        # Show results
                        summary = format_result_summary(evaluation)
                        
                        if evaluation['perfect_match']:
                            st.success(summary)
                            st.balloons()
                        else:
                            st.info(summary)
                        
                        # Detailed breakdown
                        with st.expander("📊 Detailed Breakdown"):
                            breakdown_df = pd.DataFrame(evaluation['results_per_combination'])
                            breakdown_df.columns = ['Rank', 'Correct Matches', 'Accuracy (%)']
                            st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
                        
                        # Monte Carlo simulation
                        st.write("---")
                        st.subheader("🎲 Monte Carlo Simülasyonu")
                        
                        with st.spinner("Simülasyon çalıştırılıyor..."):
                            from src.combination_evaluator import simulate_perfect_match, format_monte_carlo_summary
                            
                            # Run simulation
                            monte_carlo_result = simulate_perfect_match(
                                predictions, 
                                actual_results,
                                max_attempts=500000
                            )
                            
                            # Always save Monte Carlo result (success or failure)
                            db.save_monte_carlo_result(
                                batch_id=selected_batch,
                                attempts=monte_carlo_result['attempts'],
                                theoretical_prob=monte_carlo_result['theoretical_probability'],
                                expected_attempts=monte_carlo_result['expected_attempts'],
                                simulation_time=monte_carlo_result['simulation_time'],
                                is_success=monte_carlo_result['success'],
                                best_correct=monte_carlo_result.get('best_correct'),
                                best_attempt=monte_carlo_result.get('best_attempt'),
                                total_matches=len(actual_results)
                            )
                            
                            # Display result
                            monte_summary = format_monte_carlo_summary(monte_carlo_result)
                            st.info(monte_summary)
                            
                            if not monte_carlo_result['success']:
                                best_c = monte_carlo_result.get('best_correct', 0)
                                best_a = monte_carlo_result.get('best_attempt', 0)
                                total  = len(actual_results)
                                st.warning(
                                    f"🎯 En iyi skor: **{best_c}/{total}** doğru — "
                                    f"{best_a:,}. denemede ulaşıldı"
                                )
                            
                            # Additional details
                            with st.expander("🔍 Detaylar"):
                                st.write(f"**Simülasyon süresi:** {monte_carlo_result['simulation_time']:.2f} saniye")
                                st.write(f"**Maksimum deneme limiti:** 500,000")
                                
                                if monte_carlo_result['success']:
                                    st.write(f"**Başarı durumu:** ✅ Tam isabet bulundu!")
                                else:
                                    st.write(f"**Başarı durumu:** ⚠️ Limit aşıldı (olasılık çok düşük)")
                        
                        # Reset re-evaluate flag
                        if reeval_key in st.session_state:
                            st.session_state[reeval_key] = False
                        st.success("✅ Results saved! Check the 'Historical Performance' tab.")
            else:
                st.warning(f"No predictions or combinations found for batch: {selected_batch}")
                
        except Exception as e:
            st.error(f"Error loading batches: {e}")
            st.info("Make sure you have generated predictions first: `python main.py --mode predict`")


def show_data_overview():
    """Display data overview and statistics."""
    st.header("📚 Data Overview")
    
    # Historical matches
    historical = get_historical_matches()
    
    if historical.empty:
        st.warning("No historical data found.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", len(historical))
    with col2:
        unique_teams = len(set(historical['Ev Sahibi Takım'].tolist() + 
                               historical['Deplasman Takımı'].tolist()))
        st.metric("Unique Teams", unique_teams)
    with col3:
        result_dist = historical['Mac Sonucu'].value_counts()
        most_common = result_dist.index[0]
        st.metric("Most Common Result", most_common)
    
    # Result distribution
    st.subheader("📊 Result Distribution")
    result_counts = historical['Mac Sonucu'].value_counts()
    fig = px.pie(
        values=result_counts.values,
        names=result_counts.index,
        title="Distribution of Match Results"
    )
    st.plotly_chart(fig, width='stretch')
    
    # Team statistics
    st.subheader("🏆 Team Statistics")
    
    # Most frequent home teams
    home_teams = historical['Ev Sahibi Takım'].value_counts().head(10)
    fig = px.bar(
        x=home_teams.values,
        y=home_teams.index,
        orientation='h',
        title="Top 10 Most Frequent Home Teams",
        labels={'x': 'Number of Matches', 'y': 'Team'}
    )
    st.plotly_chart(fig, width='stretch')
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(historical.head(20), width='stretch', hide_index=True)


if __name__ == "__main__":
    main()

