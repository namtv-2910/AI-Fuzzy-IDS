import sys
sys.path.insert(0, 'src')
from fuzzy_ids import FuzzyIDS

ids = FuzzyIDS()
print("=== TEST FUZZY IDS ===\n")

test_cases = [
    ("Binh thuong", 50,   2,  20,  100),
    ("Nghi ngo",   400,  30, 400,  500),
    ("Tan cong",   900,  80, 900,  950),
]

for name, pr, fl, ps, tv in test_cases:
    r = ids.predict(pr, fl, ps, tv)
    print(f"[{name}]")
    print(f"  Input : packet={pr}, login={fl}, port={ps}, traffic={tv}")
    print(f"  Output: [{r['level'].upper()}] {r['label']} | Score={r['score']:.1f}\n")

print(">>> Fuzzy IDS hoat dong tot!")
