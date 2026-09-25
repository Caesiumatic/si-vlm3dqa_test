"""List or selectively download released video members using HTTP byte ranges."""
import argparse
import io
import json
import pathlib
import urllib.request
import zipfile


class HTTPFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.start, self.buffer = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if not n:
            return b""
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.buffer)):
            end = min(self.size - 1, self.pos + max(n, 1024 * 1024) - 1)
            sep = "&" if "?" in self.url else "?"
            request = urllib.request.Request(
                self.url + f"{sep}range={self.pos}-{end}",
                headers={"Range": f"bytes={self.pos}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"Server ignored range: {response.status}")
                self.buffer = response.read()
                self.start = self.pos
        output = self.buffer[self.pos - self.start:self.pos - self.start + n]
        self.pos += len(output)
        return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    with zipfile.ZipFile(HTTPFile(args.url, args.size)) as archive:
        entries = [{"name": i.filename, "bytes": i.file_size, "compressed": i.compress_size}
                   for i in archive.infolist()]
        pathlib.Path(args.index).write_text(json.dumps(entries, indent=2))
        videos = [i for i in archive.infolist() if i.filename.lower().endswith((".mp4", ".avi", ".mov"))]
        print(json.dumps({"members": len(entries), "videos": len(videos),
                          "video_bytes": sum(i.file_size for i in videos),
                          "examples": [i.filename for i in videos[:15]]}), flush=True)
        if args.output:
            root = pathlib.Path(args.output).resolve()
            for i in videos[:args.limit or len(videos)]:
                destination = (root / i.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError("Archive path escapes output directory")
                if destination.exists() and destination.stat().st_size == i.file_size:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(i))
                print(i.filename, i.file_size, flush=True)


if __name__ == "__main__":
    main()
