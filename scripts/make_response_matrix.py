# scripts/make_response_matrix.py
import re, csv, sys
from pathlib import Path

def parse_blocks(text):
    # Very simple heuristic: split on lines that look like "Reviewer" or numbered bullets.
    lines = text.splitlines()
    blocks = []
    current = []
    rid = "R?"
    for ln in lines:
        if re.match(r'^\s*Reviewer\s*\d+', ln, flags=re.I):
            # start new reviewer
            if current:
                blocks.append((rid, "\n".join(current).strip()))
                current = []
            rid = re.findall(r'(\d+)', ln)[0]
        elif re.match(r'^\s*(\(?[0-9ivx]+\)?[\.\)])\s+', ln, flags=re.I):
            if current:
                blocks.append((rid, "\n".join(current).strip()))
                current = []
            current.append(ln.strip())
        else:
            current.append(ln.strip())
    if current:
        blocks.append((rid, "\n".join(current).strip()))
    return blocks

def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("All Referee Reports.txt")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("earthquakes_pack/tables/referee_response_matrix.csv")
    txt = in_path.read_text(encoding="utf-8", errors="ignore")
    blocks = parse_blocks(txt)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["referee","comment_id","comment_text","action_taken","location_in_ms","status"])
        for i,(rid, blk) in enumerate(blocks, start=1):
            comment_id = f"{rid}-{i}"
            w.writerow([rid, comment_id, blk, "", "", "addressed"])
    print(f"Wrote {out_path} with {len(blocks)} rows.")

if __name__ == "__main__":
    main()
