"""Simple batch analyzer - shows which combination won."""
import sqlite3
import sys

batch_id = sys.argv[1] if len(sys.argv) > 1 else '2026-W04'
db_path = 'data/matches.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get combinations
cursor.execute("SELECT combination_rank, predictions, confidence_score FROM prediction_combinations WHERE batch_id = ? ORDER BY combination_rank", (batch_id,))
combos = cursor.fetchall()

# Get predictions with actual results  
cursor.execute("SELECT predicted_result, actual_result FROM predictions WHERE batch_id = ? ORDER BY id", (batch_id,))
preds = cursor.fetchall()

print(f'\n{"="*60}')
print(f'  {batch_id} - Kombinasyon Analizi')
print(f'{"="*60}\n')

if not combos:
    print(f'❌ {batch_id} için kombinasyon bulunamadı.')
    conn.close()
    sys.exit(0)

# Show combinations
print(f'📋 Kombinasyonlar ({len(combos)} adet):')
for rank, pred_str, score in combos:
    print(f'  #{rank}: {pred_str}')

# Check if we have actual results
actual_results = [a for p, a in preds if a]

if not actual_results:
    print(f'\n⚠️ Gerçek sonuçlar henüz girilmemiş.')
    print(f'Dashboard\'da "Enter Results" sekmesinden girebilirsiniz.')
    conn.close()
    sys.exit(0)

print(f'\n✅ Gerçek sonuçlar: {" ".join(actual_results)}')

# Evaluate each combination
print(f'\n🎯 SONUÇ TABLOSU:')
print(f'{"="*60}')
print(f'{"Rank":<8}{"Doğru":<12}{"Accuracy":<15}')
print(f'{"-"*60}')

best_rank = None
best_correct = 0

for rank, pred_str, score in combos:
    predictions = pred_str.split()
    correct = sum(1 for p, a in zip(predictions, actual_results) if p == a)
    accuracy = (correct / len(actual_results)) * 100
    
    if correct > best_correct:
        best_correct = correct
        best_rank = rank
    
    marker = '🏆' if rank == best_rank and correct == best_correct else '  '
    perfect = '' if correct < len(actual_results) else ' ✨ PERFECT!'
    print(f'{marker} #{rank:<5}{correct}/{len(actual_results):<10}{accuracy:.2f}%{perfect}')

print(f'{"="*60}')
if best_correct == len(actual_results):
    print(f'🎉 TAM İSABET: #{best_rank}. kombinasyon!')
else:
    print(f'🥇 EN YAKIN: #{best_rank}. kombinasyon ({best_correct}/{len(actual_results)} doğru - {best_correct/len(actual_results)*100:.1f}%)')
print(f'{"="*60}\n')

conn.close()
