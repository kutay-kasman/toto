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
from pathlib import Path

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
        ["This Week's Predictions", "Historical Accuracy", "Deep Analysis", "Model Statistics", "Data Overview"]
    )
    
    if page == "This Week's Predictions":
        show_predictions()
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

