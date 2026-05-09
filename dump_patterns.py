import sys
from patterns.registry import PatternRegistry

def run():
    r = PatternRegistry()
    patterns = r.detectors
    count = 0
    groups = {}
    for p in patterns:
        groups.setdefault(p.category, []).append(p.name)
        count += 1
    
    print(f"Total Patterns: {count}\n")
    for cat, list_p in groups.items():
        print(f"=== {cat.upper()} ({len(list_p)}) ===")
        for p in list_p:
            print(f"  {p}")

if __name__ == '__main__':
    run()
