"""热词只从词表生成，不另维护一份名单。"""

from pathlib import Path


def alias_lines(vocab):
    lines = []
    seen = set()
    for key in ("objects", "slots"):
        for item in vocab[key]:
            for alias in item["aliases"]:
                if alias and alias not in seen:
                    seen.add(alias)
                    lines.append(alias)
    return lines


def write_hotwords(vocab, path):
    text = "\n".join(alias_lines(vocab)) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return alias_lines(vocab)
