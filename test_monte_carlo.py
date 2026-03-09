"""Test Monte Carlo simulation with real data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import MatchDatabase
from src.combination_evaluator import simulate_perfect_match, format_monte_carlo_summary

# Test with batch 2026-W04
batch_id = '2026-W04'
db = MatchDatabase()

# Get predictions and results
predictions = db.get_predictions_by_batch(batch_id)
print(f"\n{'='*60}")
print(f"Testing Monte Carlo Simulation - {batch_id}")
print(f"{'='*60}\n")

if predictions.empty:
    print(f"❌ No predictions found for {batch_id}")
else:
    # Get actual results
    actual_results = predictions['actual_result'].dropna().tolist()
    
    if not actual_results:
        print(f"⚠️ No actual results for {batch_id}")
        print("Enter results in the dashboard first.")
    else:
        print(f"Maç sayısı: {len(actual_results)}")
        print(f"Gerçek sonuçlar: {' '.join(actual_results)}\n")
        
        # Run simulation
        result = simulate_perfect_match(predictions, actual_results, max_attempts=100000)
        
        # Show summary
        summary = format_monte_carlo_summary(result)
        print(f"\n{summary}\n")
        
        print(f"{'='*60}")
