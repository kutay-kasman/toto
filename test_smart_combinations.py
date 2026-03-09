"""Test the new smart combination generator with entropy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import MatchDatabase
import pandas as pd

# Import from dashboard
from dashboard import generate_prediction_combinations, calculate_entropy

db = MatchDatabase()

# Get a batch
batch_id = '2026-W04'
predictions = db.get_predictions_by_batch(batch_id)

print(f"\n{'='*70}")
print(f"  Testing Smart Top-K Entropy-Based Combination Generation")
print(f"{'='*70}\n")

if predictions.empty:
    print(f"❌ No predictions found for {batch_id}")
else:
    print(f"Batch: {batch_id}")
    print(f"Matches: {len(predictions)}\n")
    
    # Show entropy for each match
    print("📊 Match Uncertainties (Entropy):\n")
    print(f"{'#':<4}{'Match':<50}{'Entropy':<12}{'Type'}")
    print("-" * 70)
    
    for idx, (_, row) in enumerate(predictions.iterrows(), 1):
        entropy = calculate_entropy(
            row['probability_1'],
            row['probability_x'],
            row['probability_2']
        )
        
        match_str = f"{row['home_team']} vs {row['away_team']}"
        
        if entropy > 1.3:
            match_type = "⚠️ BELIRSIZ"
        elif entropy > 1.0:
            match_type = "🟡 Orta"
        else:
            match_type = "✅ Kesin"
        
        print(f"{idx:<4}{match_str:<50}{entropy:<12.3f}{match_type}")
    
    # Generate combinations
    print(f"\n{'='*70}")
    print("  Generating Smart Combinations...")
    print(f"{'='*70}\n")
    
    combinations = generate_prediction_combinations(predictions, n_combinations=7, top_k=5)
    
    print(f"Generated {len(combinations)} combinations\n")
    
    # Show combinations
    print("📋 Generated Combinations:\n")
    for i, combo in enumerate(combinations, 1):
        pred_str = ' '.join(combo['predictions'])
        strategy = combo.get('strategy', 'unknown')
        print(f"  #{i}: {pred_str}")
        print(f"       Score: {combo['score']*100:.6f}% | Strategy: {strategy}\n")
    
    print(f"{'='*70}")
    print("✅ Test complete!")
    print(f"{'='*70}\n")
