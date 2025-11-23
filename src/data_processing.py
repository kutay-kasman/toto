"""
Data processing and feature engineering module.
Handles cleaning, transformation, and feature creation.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data cleaning and feature engineering."""
    
    def __init__(self, lookback_games: int = 10):
        """
        Initialize data processor.
        
        Args:
            lookback_games: Number of previous games to consider for statistics
        """
        self.lookback_games = lookback_games
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw match data.
        
        Args:
            df: Raw DataFrame with match data
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning {len(df)} records")
        
        # Remove rows with missing match names
        df = df.dropna(subset=['Karsilasma'] if 'Karsilasma' in df.columns else ['Ev Sahibi Takım'])
        
        # If we have 'Karsilasma' column, split it
        if 'Karsilasma' in df.columns:
            df['Karsilasma'] = df['Karsilasma'].str.strip()
            split_df = df['Karsilasma'].str.split('-', expand=True)
            
            if split_df.shape[1] >= 2:
                df['Ev Sahibi Takım'] = split_df[0].str.strip()
                df['Deplasman Takımı'] = split_df[1].str.strip()
            else:
                df['Ev Sahibi Takım'] = np.nan
                df['Deplasman Takımı'] = np.nan
            
            df = df.dropna(subset=['Ev Sahibi Takım', 'Deplasman Takımı'])
        
        # Ensure we have required columns
        required_cols = ['Ev Sahibi Takım', 'Deplasman Takımı', 'Mac Sonucu']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
        
        # Filter valid results
        df = df[df['Mac Sonucu'].isin(['1', 'X', '2'])]
        
        # Select and order columns
        df = df[required_cols].copy()
        
        logger.info(f"Cleaned data: {len(df)} records remaining")
        return df.reset_index(drop=True)
    
    def calculate_performance_ratios(self, df: pd.DataFrame, team: str, 
                                     current_index: int) -> Tuple[float, float, float]:
        """
        Calculate win/draw/loss ratios for a team.
        
        Args:
            df: Historical match data
            team: Team name
            current_index: Current match index (to avoid data leakage)
            
        Returns:
            Tuple of (win_ratio, draw_ratio, loss_ratio)
        """
        past_games = df.iloc[:current_index]
        
        team_games = past_games[
            (past_games['Ev Sahibi Takım'] == team) | 
            (past_games['Deplasman Takımı'] == team)
        ].tail(self.lookback_games)
        
        if team_games.empty:
            return 0.0, 0.0, 0.0
        
        total_wins = 0
        total_draws = 0
        total_losses = 0
        
        for _, game in team_games.iterrows():
            match_result = game['Mac Sonucu']
            is_home = (game['Ev Sahibi Takım'] == team)
            
            if match_result == 'X':
                total_draws += 1
            elif (is_home and match_result == '1') or (not is_home and match_result == '2'):
                total_wins += 1
            else:
                total_losses += 1
        
        valid_lookback = len(team_games)
        win_ratio = total_wins / valid_lookback
        draw_ratio = total_draws / valid_lookback
        loss_ratio = total_losses / valid_lookback
        
        return win_ratio, draw_ratio, loss_ratio
    
    def calculate_goals_statistics(self, df: pd.DataFrame, team: str,
                                   current_index: int) -> Tuple[float, float]:
        """
        Calculate average goals scored and conceded for a team.
        
        Args:
            df: Historical match data (must have Home_Goals and Away_Goals columns)
            team: Team name
            current_index: Current match index (to avoid data leakage)
            
        Returns:
            Tuple of (avg_goals_scored, avg_goals_conceded)
        """
        past_games = df.iloc[:current_index]
        
        team_games = past_games[
            (past_games['Ev Sahibi Takım'] == team) | 
            (past_games['Deplasman Takımı'] == team)
        ].tail(self.lookback_games)
        
        if team_games.empty:
            return 0.0, 0.0
        
        total_goals_scored = 0
        total_goals_conceded = 0
        valid_games = 0
        
        for _, game in team_games.iterrows():
            is_home = (game['Ev Sahibi Takım'] == team)
            
            # Get goals if available
            home_goals = game.get('Home_Goals', None)
            away_goals = game.get('Away_Goals', None)
            
            # Only use matches with actual goal data
            if pd.notna(home_goals) and pd.notna(away_goals):
                if is_home:
                    total_goals_scored += int(home_goals)
                    total_goals_conceded += int(away_goals)
                else:
                    total_goals_scored += int(away_goals)
                    total_goals_conceded += int(home_goals)
                valid_games += 1
            # If goals not available, skip this match (don't estimate)
            # This ensures we only use real data
        
        if valid_games == 0:
            return 0.0, 0.0
        
        avg_goals_scored = total_goals_scored / valid_games
        avg_goals_conceded = total_goals_conceded / valid_games
        
        return avg_goals_scored, avg_goals_conceded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features for training.
        
        Args:
            df: Cleaned match data
            
        Returns:
            DataFrame with added feature columns
        """
        logger.info("Creating statistical features")
        
        df = df.sort_index().reset_index(drop=True)
        
        # Initialize feature columns
        df['Home_Win_Ratio'] = 0.0
        df['Home_Draw_Ratio'] = 0.0
        df['Home_Loss_Ratio'] = 0.0
        df['Away_Win_Ratio'] = 0.0
        df['Away_Draw_Ratio'] = 0.0
        df['Away_Loss_Ratio'] = 0.0
        
        # Initialize goal-based features
        df['Home_Avg_Goals_Scored'] = 0.0
        df['Home_Avg_Goals_Conceded'] = 0.0
        df['Away_Avg_Goals_Scored'] = 0.0
        df['Away_Avg_Goals_Conceded'] = 0.0
        
        # Calculate features for each match
        for index, row in df.iterrows():
            home_team = row['Ev Sahibi Takım']
            away_team = row['Deplasman Takımı']
            
            # Home team statistics
            h_win, h_draw, h_loss = self.calculate_performance_ratios(df, home_team, index)
            df.loc[index, 'Home_Win_Ratio'] = h_win
            df.loc[index, 'Home_Draw_Ratio'] = h_draw
            df.loc[index, 'Home_Loss_Ratio'] = h_loss
            
            # Away team statistics
            a_win, a_draw, a_loss = self.calculate_performance_ratios(df, away_team, index)
            df.loc[index, 'Away_Win_Ratio'] = a_win
            df.loc[index, 'Away_Draw_Ratio'] = a_draw
            df.loc[index, 'Away_Loss_Ratio'] = a_loss
            
            # Goal-based statistics
            h_goals_scored, h_goals_conceded = self.calculate_goals_statistics(df, home_team, index)
            df.loc[index, 'Home_Avg_Goals_Scored'] = h_goals_scored
            df.loc[index, 'Home_Avg_Goals_Conceded'] = h_goals_conceded
            
            a_goals_scored, a_goals_conceded = self.calculate_goals_statistics(df, away_team, index)
            df.loc[index, 'Away_Avg_Goals_Scored'] = a_goals_scored
            df.loc[index, 'Away_Avg_Goals_Conceded'] = a_goals_conceded
        
        # Remove rows without enough history
        df = df.iloc[self.lookback_games:].reset_index(drop=True)
        
        logger.info(f"Features created for {len(df)} matches")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepare data for ML training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (X_features, y_target, feature_columns)
        """
        from sklearn.preprocessing import LabelEncoder
        
        logger.info("Preparing training data")
        
        # Encode target variable
        le = LabelEncoder()
        df['Mac Sonucu_Encoded'] = le.fit_transform(df['Mac Sonucu'])
        
        # One-hot encode team names
        df_home = pd.get_dummies(df['Ev Sahibi Takım'], prefix='Ev')
        df_away = pd.get_dummies(df['Deplasman Takımı'], prefix='Dep')
        
        # Combine features
        stat_cols = [
            'Home_Win_Ratio', 'Home_Draw_Ratio', 'Home_Loss_Ratio',
            'Away_Win_Ratio', 'Away_Draw_Ratio', 'Away_Loss_Ratio',
            'Home_Avg_Goals_Scored', 'Home_Avg_Goals_Conceded',
            'Away_Avg_Goals_Scored', 'Away_Avg_Goals_Conceded'
        ]
        
        X = pd.concat([df_home, df_away, df[stat_cols]], axis=1)
        y = df['Mac Sonucu_Encoded']
        
        feature_columns = X.columns.tolist()
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y, feature_columns

