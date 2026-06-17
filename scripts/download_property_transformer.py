from pathlib import Path
from urllib.request import Request, urlopen
import time


URL = "https://zenodo.org/api/records/17665048/files/Property_Transformer.pt/content"
OUT = Path("/data/ckpt/Property_Transformer.pt")
CHUNK_SIZE = 1024 * 1024


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.stat().st_size > 0:
        print(f"Removing partial/existing file: {OUT} ({OUT.stat().st_size / 1024**3:.2f} GiB)", flush=True)
        OUT.unlink()

    for attempt in range(1, 6):
        try:
            req = Request(URL, headers={"User-Agent": "mmpae-download"})
            with urlopen(req, timeout=120) as response, open(OUT, "wb") as handle:
                total = response.headers.get("Content-Length")
                total = int(total) if total else None
                done = 0
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r{done / 1024**3:.2f}/{total / 1024**3:.2f} GiB", end="", flush=True)
                    else:
                        print(f"\r{done / 1024**3:.2f} GiB", end="", flush=True)
                print()
            print(f"Downloaded: {OUT} ({OUT.stat().st_size / 1024**3:.2f} GiB)", flush=True)
            return
        except Exception as exc:
            print(f"attempt {attempt} failed: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(10)


if __name__ == "__main__":
    main()
