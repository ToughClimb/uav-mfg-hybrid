"""Serve the generated experiment reports on a local loopback address."""

import argparse
import functools
import http.server
import json
import os
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--directory",type=Path,default=Path("results"))
    args=parser.parse_args()
    directory=args.directory.resolve()
    if not (directory/"eswa_verified/report.html").exists():
        raise FileNotFoundError(directory/"eswa_verified/report.html")
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1",args.port),handler) as server:
        url=f"http://127.0.0.1:{args.port}/eswa_verified/report.html"
        metadata={"pid":os.getpid(),"command":[sys.executable,*sys.argv],"venv_prefix":sys.prefix,"url":url}
        (directory/"preview_process.json").write_text(json.dumps(metadata,indent=2)+'\n')
        print(url,flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Preview stopped",flush=True)


if __name__=="__main__":
    main()
