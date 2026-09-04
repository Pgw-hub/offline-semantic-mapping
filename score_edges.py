import json, sys, collections
gt = json.load(open(sys.argv[1])); pred = json.load(open(sys.argv[2]))
ok = 0; rows = []
for e in pred:
    g = gt.get(str(e["pair_id"])); p = e.get("relation")
    ok += (g == p); rows.append(f'{e["pair_id"]:02d} {e["object_1_tag"]}↔{e["object_2_tag"]}: GT={g:<12} pred={p:<12} {"✓" if g==p else "✗"}')
print("\n".join(rows)); print(f"정확도 {ok}/{len(pred)} = {ok/len(pred)*100:.0f}%")
print("예측 분포:", collections.Counter(e.get("relation") for e in pred).most_common())
