import argparse
import json
from pathlib import Path
from urllib.request import urlopen


DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/digital-go-jp/lawqa_jp/main/data/selection.json"
)
DEFAULT_OUTPUT_PATH = Path("evals/lawqa_jp_20.json")


def download_lawqa_samples(
    source_url: str = DEFAULT_SOURCE_URL,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int = 20,
) -> dict:
    with urlopen(source_url) as response:
        raw_data = json.loads(response.read().decode("utf-8"))

    samples = raw_data["samples"][:limit]
    output = {
        "source_url": source_url,
        "limit": limit,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a local lawqa_jp eval subset.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    output = download_lawqa_samples(
        source_url=args.source_url,
        output_path=Path(args.output),
        limit=args.limit,
    )
    print(f"saved: {args.output}")
    print(f"samples: {len(output['samples'])}")


if __name__ == "__main__":
    main()
