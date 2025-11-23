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


@st.cache_resource
def get_db_connection():
    """Get direct SQLite database connection (cached)."""
    db_path = Path("data/matches.db")
    if not db_path.exists():
        # Try alternative path
        db_path = Path("matches.db")
    
    if not db_path.exists():
        st.error(f"Database not found at {db_path}. Please ensure data/matches.db exists.")
        return None
    
    return sqlite3.connect(str(db_path))


@st.cache_data
def get_historical_matches():
    """Get historical matches from database."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT home_team as 'Ev Sahibi Takım',
                   away_team as 'Deplasman Takımı',
                   result as 'Mac Sonucu',
                   home_goals as 'Home_Goals',
                   away_goals as 'Away_Goals'
            FROM historical_matches
            ORDER BY match_date, id
        """, conn)
        return df
    except Exception as e:
        st.error(f"Error loading historical matches: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data
def get_upcoming_matches():
    """Get upcoming matches from database."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT home_team as 'Ev Sahibi Takım',
                   away_team as 'Deplasman Takımı',
                   match_date
            FROM upcoming_matches
            ORDER BY match_date, id
        """, conn)
        return df
    except Exception as e:
        st.error(f"Error loading upcoming matches: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data
def get_predictions(limit=1000):
    """Get predictions from database."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT home_team, away_team, predicted_result,
                   probability_1, probability_x, probability_2,
                   actual_result, is_correct, prediction_date
            FROM predictions
            ORDER BY prediction_date DESC
            LIMIT ?
        """, conn, params=(limit,))
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data
def get_prediction_accuracy():
    """Calculate overall prediction accuracy."""
    conn = get_db_connection()
    if conn is None:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}
    
    try:
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
        st.error(f"Error calculating accuracy: {e}")
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}
    finally:
        conn.close()


def main():
    """Main dashboard function."""
    st.markdown('<h1 class="main-header">⚽ Football Match Prediction Dashboard</h1>', 
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
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["This Week's Predictions", "Historical Accuracy", "Model Statistics", "Data Overview"]
    )
    
    if page == "This Week's Predictions":
        show_predictions()
    elif page == "Historical Accuracy":
        show_accuracy()
    elif page == "Model Statistics":
        show_model_stats()
    elif page == "Data Overview":
        show_data_overview()


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
            st.dataframe(df_results, use_container_width=True, hide_index=True)
            
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
            st.plotly_chart(fig, use_container_width=True)
            
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
            st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
    
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent predictions table
    st.subheader("📋 Recent Predictions")
    display_cols = ['home_team', 'away_team', 'predicted_result', 
                   'actual_result', 'is_correct', 'prediction_date']
    st.dataframe(
        completed[display_cols].head(20),
        use_container_width=True,
        hide_index=True
    )


def show_model_stats():
    """Display model statistics and features."""
    st.header("🤖 Model Statistics")
    
    conn = get_db_connection()
    if conn is None:
        return
    
    try:
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
            st.plotly_chart(fig, use_container_width=True)
            
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
    finally:
        conn.close()


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
    st.plotly_chart(fig, use_container_width=True)
    
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(historical.head(20), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

