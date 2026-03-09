"""
Combination evaluator module for comparing predictions against actual results.
"""

import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def evaluate_combinations(combinations: List[dict], actual_results: List[str]) -> Dict:
    """
    Compare all combinations against actual results.
    
    Args:
        combinations: List of dicts with 'predictions' (list) and 'score' (float)
        actual_results: List of actual results (e.g., ['1', 'X', '2', '1', ...])
    
    Returns:
        {
            'winning_rank': 2,          # Which combination matched perfectly (None if none)
            'best_rank': 1,             # Which combination had most correct
            'best_correct': 10,         # Number of correct matches in best combination
            'total_matches': 12,        # Total number of matches
            'perfect_match': True,      # Whether any combination was 100% correct
            'results_per_combination': [ # Detailed results for each
                {'rank': 1, 'correct': 8, 'accuracy': 66.67},
                {'rank': 2, 'correct': 10, 'accuracy': 83.33},
                ...
            ]
        }
    """
    if not combinations or not actual_results:
        return {
            'winning_rank': None,
            'best_rank': None,
            'best_correct': 0,
            'total_matches': 0,
            'perfect_match': False,
            'results_per_combination': []
        }
    
    total_matches = len(actual_results)
    results_per_combination = []
    winning_rank = None
    best_rank = None
    best_correct = 0
    
    # Evaluate each combination
    for combo in combinations:
        rank = combinations.index(combo) + 1
        predictions = combo['predictions']
        
        # Count correct predictions
        correct = sum(1 for pred, actual in zip(predictions, actual_results) 
                     if pred == actual)
        accuracy = (correct / total_matches * 100) if total_matches > 0 else 0
        
        results_per_combination.append({
            'rank': rank,
            'correct': correct,
            'accuracy': accuracy
        })
        
        # Track best
        if correct > best_correct:
            best_correct = correct
            best_rank = rank
        
        # Check if perfect
        if correct == total_matches:
            winning_rank = rank
            logger.info(f"Perfect match found at combination #{rank}")
    
    perfect_match = winning_rank is not None
    
    result = {
        'winning_rank': winning_rank,
        'best_rank': best_rank,
        'best_correct': best_correct,
        'total_matches': total_matches,
        'perfect_match': perfect_match,
        'results_per_combination': results_per_combination
    }
    
    logger.info(f"Evaluation complete: best={best_rank} ({best_correct}/{total_matches}), "
                f"perfect={'Yes' if perfect_match else 'No'}")
    
    return result


def format_result_summary(evaluation: Dict) -> str:
    """
    Create a human-readable summary of evaluation results.
    
    Args:
        evaluation: Result from evaluate_combinations()
    
    Returns:
        Formatted string summary
    """
    if evaluation['perfect_match']:
        return (f"🎯 TAM İSABET! {evaluation['winning_rank']}. kombinasyon "
                f"{evaluation['total_matches']}/{evaluation['total_matches']} doğru!")
    else:
        return (f"En yakın: {evaluation['best_rank']}. kombinasyon "
                f"{evaluation['best_correct']}/{evaluation['total_matches']} doğru "
                f"({evaluation['results_per_combination'][evaluation['best_rank']-1]['accuracy']:.1f}%)")


def match_predictions_to_results(predictions_df, actual_results_dict: Dict[Tuple[str, str], str]) -> List[str]:
    """
    Convert actual results dictionary to ordered list matching predictions DataFrame.
    
    Args:
        predictions_df: DataFrame with columns 'home_team', 'away_team'
        actual_results_dict: Dict mapping (home, away) tuples to results
    
    Returns:
        Ordered list of results matching predictions order
    """
    results = []
    for _, row in predictions_df.iterrows():
        key = (row['home_team'], row['away_team'])
        result = actual_results_dict.get(key, None)
        if result:
            results.append(result)
        else:
            logger.warning(f"No result found for {key[0]} vs {key[1]}")
    
    return results


def simulate_perfect_match(predictions_df, actual_results: List[str], max_attempts: int = 100000) -> Dict:
    """
    Monte Carlo simulation: How many random attempts (based on probabilities) 
    until we achieve a perfect match?
    
    This answers: "Kaçıncı denemede tüm maçları doğru bilirdik?"
    
    Args:
        predictions_df: DataFrame with probability_1, probability_x, probability_2 columns
        actual_results: List of actual results ['1', 'X', '2', ...]
        max_attempts: Maximum simulation attempts (safety limit)
    
    Returns:
        {
            'success': bool,               # Did we find perfect match?
            'attempts': int,               # Number of attempts to perfect match
            'theoretical_probability': float,  # P(all correct) = ∏ P(each correct)
            'expected_attempts': int,      # 1 / theoretical_probability
            'simulation_time': float,      # Time in seconds
            'match_probabilities': list    # P(correct) for each match
        }
    """
    import numpy as np
    import time
    
    start_time = time.time()
    
    # Extract probabilities for each match
    match_probabilities = []
    prob_matrix = []  # List of [p1, px, p2] for each match
    
    for _, row in predictions_df.iterrows():
        p1 = row.get('probability_1', 0)
        px = row.get('probability_x', 0)
        p2 = row.get('probability_2', 0)
        
        # Normalize to ensure sum = 1.0
        total = p1 + px + p2
        if total > 0:
            p1 = p1 / total
            px = px / total
            p2 = p2 / total
        else:
            # Default to equal probabilities
            p1, px, p2 = 1/3, 1/3, 1/3
        
        prob_matrix.append([p1, px, p2])
    
    # Calculate probability of each match being correct
    for i, actual in enumerate(actual_results):
        p1, px, p2 = prob_matrix[i]
        
        if actual == '1':
            match_probabilities.append(p1)
        elif actual == 'X':
            match_probabilities.append(px)
        elif actual == '2':
            match_probabilities.append(p2)
        else:
            logger.warning(f"Unknown result: {actual}")
            match_probabilities.append(0.33)  # Default
    
    # Calculate theoretical probability of perfect match
    theoretical_probability = np.prod(match_probabilities)
    expected_attempts = int(1 / theoretical_probability) if theoretical_probability > 0 else float('inf')
    
    logger.info(f"Theoretical probability: {theoretical_probability:.6f} ({theoretical_probability*100:.3f}%)")
    logger.info(f"Expected attempts: {expected_attempts:,}")
    
    # Run Monte Carlo simulation
    best_correct = 0
    best_attempt = 0
    best_prediction = None
    
    for attempt in range(1, max_attempts + 1):
        # Generate random prediction based on probabilities
        simulated_prediction = []
        
        for probs in prob_matrix:
            # Weighted random choice
            outcome = np.random.choice(['1', 'X', '2'], p=probs)
            simulated_prediction.append(outcome)
        
        # Count correct matches
        correct = sum(1 for pred, actual in zip(simulated_prediction, actual_results) if pred == actual)
        
        # Track best score
        if correct > best_correct:
            best_correct = correct
            best_attempt = attempt
            best_prediction = simulated_prediction.copy()
            logger.info(f"New best: {best_correct}/{len(actual_results)} at attempt #{attempt:,}")
        
        # Check if perfect match
        if simulated_prediction == actual_results:
            simulation_time = time.time() - start_time
            
            logger.info(f"🎯 Perfect match found at attempt #{attempt:,}!")
            logger.info(f"Simulation time: {simulation_time:.3f}s")
            
            return {
                'success': True,
                'attempts': attempt,
                'theoretical_probability': float(theoretical_probability),
                'expected_attempts': expected_attempts,
                'simulation_time': simulation_time,
                'match_probabilities': match_probabilities,
                'best_correct': len(actual_results)
            }
    
    # Max attempts reached without success
    simulation_time = time.time() - start_time
    
    logger.warning(f"Max attempts ({max_attempts:,}) reached without perfect match")
    logger.warning(f"Best score: {best_correct}/{len(actual_results)} at attempt #{best_attempt:,}")
    
    return {
        'success': False,
        'attempts': max_attempts,
        'theoretical_probability': float(theoretical_probability),
        'expected_attempts': expected_attempts,
        'simulation_time': simulation_time,
        'match_probabilities': match_probabilities,
        'best_correct': best_correct,
        'best_attempt': best_attempt,
        'best_prediction': best_prediction
    }


def format_monte_carlo_summary(simulation_result: Dict) -> str:
    """
    Create a human-readable summary of Monte Carlo simulation.
    
    Args:
        simulation_result: Result from simulate_perfect_match()
    
    Returns:
        Formatted string summary
    """
    if not simulation_result['success']:
        best_correct = simulation_result.get('best_correct', 0)
        best_attempt = simulation_result.get('best_attempt', 0)
        total = len(simulation_result.get('match_probabilities', []))
        
        return (f"⚠️ Simülasyon {simulation_result['attempts']:,} denemede başarısız oldu.\n"
                f"Beklenen: {simulation_result['expected_attempts']:,} deneme\n"
                f"Teorik olasılık çok düşük: {simulation_result['theoretical_probability']*100:.4f}%\n\n"
                f"🎯 En yakın skor: {best_correct}/{total} doğru\n"
                f"📍 {best_attempt:,}. denemede ulaşıldı!")
    
    attempts = simulation_result['attempts']
    expected = simulation_result['expected_attempts']
    prob = simulation_result['theoretical_probability'] * 100
    
    # Check if lucky or unlucky
    if attempts < expected:
        diff_pct = ((expected - attempts) / expected) * 100
        luck = f"✨ Şanslı! ({diff_pct:.0f}% daha az deneme)"
    elif attempts > expected:
        diff_pct = ((attempts - expected) / expected) * 100
        luck = f"😅 Şanssız ({diff_pct:.0f}% daha fazla deneme)"
    else:
        luck = "🎯 Tam beklenen kadar!"
    
    return (f"🎲 Monte Carlo Simülasyonu\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Teorik olasılık: {prob:.3f}% (1/{expected:,})\n"
            f"Beklenen deneme: {expected:,}\n"
            f"Gerçek deneme: {attempts:,}\n"
            f"{luck}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
