"""
Interactive terminal tool to enter results and see Monte Carlo simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.database import MatchDatabase
from src.combination_evaluator import evaluate_combinations, simulate_perfect_match
import pandas as pd

def main():
    db = MatchDatabase()
    
    # Get available batches
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT batch_id FROM prediction_combinations ORDER BY batch_id DESC")
    batches = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not batches:
        print("❌ Hiç batch bulunamadı. Önce tahmin yapın: python main.py --mode predict")
        return
    
    print(f"\n{'='*60}")
    print("  🎯 Sonuç Girme ve Monte Carlo Simülasyonu")
    print(f"{'='*60}\n")
    
    # Show available batches
    print("Mevcut Batch'ler:")
    for i, batch in enumerate(batches, 1):
        print(f"  {i}. {batch}")
    
    # Select batch
    choice = input(f"\nBatch seçin (1-{len(batches)}): ")
    try:
        batch_idx = int(choice) - 1
        selected_batch = batches[batch_idx]
    except:
        print("❌ Geçersiz seçim!")
        return
    
    print(f"\n{'='*60}")
    print(f"  {selected_batch}")
    print(f"{'='*60}\n")
    
    # Get predictions and combinations
    predictions = db.get_predictions_by_batch(selected_batch)
    combinations = db.get_combinations_for_batch(selected_batch)
    
    if predictions.empty or not combinations:
        print(f"❌ {selected_batch} için veri bulunamadı!")
        return
    
    # Show matches
    print(f"Maçlar ({len(predictions)} adet):\n")
    for idx, (_, row) in enumerate(predictions.iterrows(), 1):
        print(f"  {idx:2}. {row['home_team']:<30} vs {row['away_team']:<30}")
    
    # Enter results
    print(f"\n{'='*60}")
    print("Sonuçları girin (1/X/2):")
    print(f"{'='*60}\n")
    
    actual_results = []
    for idx, (_, row) in enumerate(predictions.iterrows(), 1):
        while True:
            result = input(f"  Maç {idx} ({row['home_team']} vs {row['away_team']}): ").strip().upper()
            if result in ['1', 'X', '2']:
                actual_results.append(result)
                break
            else:
                print("    ❌ Geçersiz! 1, X veya 2 girin.")
    
    # Evaluate combinations
    print(f"\n{'='*60}")
    print("  📊 DEĞERLENDİRME")
    print(f"{'='*60}\n")
    
    evaluation = evaluate_combinations(combinations, actual_results)
    
    print(f"Toplam maç: {evaluation['total_matches']}")
    print(f"En iyi kombinasyon: #{evaluation['best_rank']} ({evaluation['best_correct']}/{evaluation['total_matches']} doğru)\n")
    
    if evaluation['perfect_match']:
        print(f"🎉 TAM İSABET! #{evaluation['winning_rank']}. kombinasyon!\n")
    else:
        print(f"⚠️ Tam isabet yok. En yakın: #{evaluation['best_rank']}\n")
    
    # Show breakdown
    print("Kombinasyon Detayları:")
    print(f"{'Rank':<8}{'Doğru':<15}{'Accuracy'}")
    print(f"{'-'*40}")
    for r in evaluation['results_per_combination']:
        marker = '🏆' if r['rank'] == evaluation['best_rank'] else '  '
        print(f"{marker} #{r['rank']:<5}{r['correct']}/{evaluation['total_matches']:<13}{r['accuracy']:.1f}%")
    
    # Monte Carlo simulation
    print(f"\n{'='*60}")
    print("  🎲 MONTE CARLO SİMÜLASYONU")
    print(f"{'='*60}\n")
    
    print("Simülasyon çalıştırılıyor... (Bu birkaç saniye sürebilir)")
    
    monte_result = simulate_perfect_match(predictions, actual_results, max_attempts=500000)
    
    if monte_result['success']:
        attempts = monte_result['attempts']
        expected = monte_result['expected_attempts']
        prob = monte_result['theoretical_probability'] * 100
        
        print(f"\n✅ Başarı! Tam isabet bulundu.\n")
        print(f"Teorik olasılık:  {prob:.4f}% (1 / {expected:,})")
        print(f"Beklenen deneme:  {expected:,}")
        print(f"Gerçek deneme:    {attempts:,}\n")
        
        if attempts < expected:
            diff_pct = ((expected - attempts) / expected) * 100
            print(f"✨ Şanslı! {diff_pct:.0f}% daha az deneme!")
        elif attempts > expected:
            diff_pct = ((attempts - expected) / expected) * 100
            print(f"😅 Şanssız... {diff_pct:.0f}% daha fazla deneme")
        else:
            print(f"🎯 Tam beklenen kadar!")
        
        print(f"\nSimülasyon süresi: {monte_result['simulation_time']:.2f} saniye")
        
        # Save to database
        db.save_combination_result(
            batch_id=selected_batch,
            winning_rank=evaluation['winning_rank'],
            total_correct=evaluation['best_correct'],
            total_matches=evaluation['total_matches']
        )
        
        db.save_monte_carlo_result(
            batch_id=selected_batch,
            attempts=monte_result['attempts'],
            theoretical_prob=monte_result['theoretical_probability'],
            expected_attempts=monte_result['expected_attempts'],
            simulation_time=monte_result['simulation_time']
        )
        
        print("\n✅ Sonuçlar veritabanına kaydedildi!")
        
    else:
        print(f"\n⚠️ 100,000 denemede tam isabet bulunamadı.")
        print(f"Teorik olasılık çok düşük: {monte_result['theoretical_probability']*100:.6f}%")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
