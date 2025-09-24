from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import orjson


def load(path: Path) -> List[dict]:
    return orjson.loads(path.read_bytes())


def get_texts(decision: dict) -> Dict[str, str]:
    d = (decision.get("decision") or "")
    r = (decision.get("reasoning") or "")
    cons = decision.get("considerations") or {}
    inf = cons.get("in_favor") or []
    ag = cons.get("against") or []
    body = (d + "\n\n" + r).strip()
    infs = "\n".join([x.strip() for x in inf if isinstance(x, str)]).strip()
    ags = "\n".join([x.strip() for x in ag if isinstance(x, str)]).strip()
    return {"body": body, "in_favor": infs, "against": ags}


def main() -> None:
    merged_path = Path("data/merged/merged_dilemmas_responses.json")
    data = load(merged_path)
    total = 0
    nonempty = 0
    empties: List[tuple] = []

    for item in data:
        item_id = int(item.get("id"))
        decs = item.get("decisions") or {}
        for model, decision in decs.items():
            texts = get_texts(decision)
            for kind, text in texts.items():
                total += 1
                if text:
                    nonempty += 1
                else:
                    empties.append((item_id, model, kind))

    print(f"total_slots={total} nonempty={nonempty} empty={len(empties)}")
    for row in empties:
        print("EMPTY:", row)


if __name__ == "__main__":
    main()


