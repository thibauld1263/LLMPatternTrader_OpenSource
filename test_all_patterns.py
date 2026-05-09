import sys
import pandas as pd
from data_fetcher import MT5DataFetcher
from patterns.registry import PatternRegistry
from config import SYMBOLS

def main():
    fetcher = MT5DataFetcher()
    registry = PatternRegistry()
    
    print("Initialisation de MT5...")
    if not fetcher.initialize():
        print("❌ Erreur de connexion à MT5. Assure-toi que MT5 est ouvert.")
        return
        
    print(f"Lancement du Diagnostic court sur 3 symboles (20 000 bougies en H1)...")
    
    pattern_counts = {}
    
    for symbol in SYMBOLS[:3]:
        print(f"➤ Récupération et Scan : {symbol}...")
        df = fetcher.fetch_data(symbol, "H1", use_cache=False, bars=20000)
        
        if df is None or df.empty:
            print(f"  ⚠️ Aucune donnée pour {symbol}")
            continue
            
        signals = registry.scan(df)
        
        # Count non-zero signals for each column
        for col in signals.columns:
            count = (signals[col] != 0).sum()
            pattern_counts[col] = pattern_counts.get(col, 0) + count
            
    print("\n==========================================")
    print("       RÉSULTATS DU SCAN DIAGNOSTIQUE       ")
    print("==========================================")
    
    # Sort by occurrences
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1])
    
    zero_patterns = [p for p in sorted_patterns if p[1] == 0]
    active_patterns = [p for p in sorted_patterns if p[1] > 0]
    
    print(f"\nTotal de patterns analysés : {len(pattern_counts)}")
    print(f"Patterns avec ZERO déclenchement : {len(zero_patterns)}")
    
    print("\n--- ❌ ZERO OCCURRENCE (À Vérifier / Cassés) ---")
    for pat, count in zero_patterns:
        print(f"  - {pat}")
        
    print("\n--- ⚠️ LES 10 PATTERNS LES PLUS RARES ---")
    for pat, count in active_patterns[:10]:
        print(f"  - {pat}: {count} déclenchements")
        
    print("\n--- ✅ LES 10 PATTERNS LES PLUS FRÉQUENTS ---")
    top_10 = list(reversed(active_patterns))[:10]
    for pat, count in top_10:
        print(f"  - {pat}: {count} déclenchements")

    print("\nDiagnostic terminé.")

if __name__ == '__main__':
    main()
