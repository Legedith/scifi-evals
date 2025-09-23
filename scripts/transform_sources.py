import json
from pathlib import Path


def split_source_field(item: dict) -> dict:
    """Split the `source` field into `source` and `author` using the last ' by '.

    If no ' by ' is found, leave `author` as an empty string and keep `source` unchanged.
    """
    src = item.get("source", "")
    # Use rpartition to split on the last occurrence of ' by '
    book, sep, author = src.rpartition(" by ")
    if sep == "":
        # No delimiter found; return item with empty author
        return {**item, "source": src, "author": ""}
    # Trim whitespace
    book = book.strip()
    author = author.strip()
    return {**item, "source": book, "author": author}


def transform_file(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("expected top-level JSON array")
    transformed = [split_source_field(item) for item in data]
    output_path.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data/source/original_dilemmas.json"
    output_path = root / "data/source/formatted_dilemmas.json"
    transform_file(input_path, output_path)
    print(f"Wrote formatted JSON to: {output_path}")


if __name__ == "__main__":
    main()


