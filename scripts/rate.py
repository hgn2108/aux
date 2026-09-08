"""Local blind-rating server for Slice 1.

Serves the pooled results with audio so relevance can actually be judged by ear, and writes
ratings to `evals/ratings.private.json`.

**What the rater is not shown, and why.** Not the filename or artist — a title gives away
genre and familiarity, which is what we are asking them to judge from the audio. Not the
rank, and not the score — a result presented first is rated higher, so position must carry
no information. Items are shuffled within each query with a fixed seed, so the order is
reproducible without being informative.

This is an evaluation tool, not the Slice 1B player: no queue, no library view, no
behavioural logging, and nothing it produces enters the product.

    python scripts/rate.py          # then open the printed URL
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
RATINGS = EVALS / "ratings.private.json"

PAGE = """<!doctype html>
<meta charset="utf-8"><title>aux — relevance rating</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--card:#1e2126;--line:#2c3036;--ok:#5ac57a}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}
 header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:16px 24px;z-index:2}
 .q{font-size:22px;font-weight:600}
 .meta{color:var(--dim);font-size:13px;margin-top:4px}
 .bar{height:4px;background:var(--line);border-radius:2px;margin-top:12px;overflow:hidden}
 .bar>div{height:100%;background:var(--ok);width:0%;transition:width .2s}
 main{padding:24px;max-width:760px;margin:0 auto}
 .item{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:12px}
 .item.current{border-color:var(--ok)}
 .row{display:flex;align-items:center;gap:12px}
 button{background:#282c33;color:var(--fg);border:1px solid var(--line);border-radius:8px;
        padding:9px 14px;font-size:15px;cursor:pointer}
 button:hover{background:#31363e}
 .rate button{width:44px;font-variant-numeric:tabular-nums}
 .rate button.sel{background:var(--ok);color:#07130b;border-color:var(--ok);font-weight:700}
 .play{width:92px}
 .spacer{flex:1}
 .hint{color:var(--dim);font-size:12px;margin-top:6px}
 footer{padding:0 24px 48px;max-width:760px;margin:0 auto;color:var(--dim);font-size:13px}
 .nav{display:flex;gap:10px;margin:20px 0}
 .done{color:var(--ok)}
</style>
<header>
  <div class="q" id="query">—</div>
  <div class="meta" id="meta"></div>
  <div class="bar"><div id="prog"></div></div>
</header>
<main>
  <div id="items"></div>
  <div class="nav">
    <button onclick="go(-1)">← Previous query</button>
    <button onclick="go(1)">Next query →</button>
    <span class="spacer"></span>
    <button onclick="save(true)">Save now</button>
  </div>
  <div class="hint">Keys: <b>J/K</b> move between clips · <b>Space</b> play/pause ·
    <b>1–5</b> rate · <b>N/P</b> next/previous query. Playback starts partway in, so you
    hear the body of the track rather than the intro.</div>
</main>
<footer>
  <p><b>1</b> = not relevant at all · <b>3</b> = partly fits · <b>5</b> = exactly what I asked for.</p>
  <p>Judge only the audio against the query. Titles, artists and ranking are hidden on
     purpose. Rate what you hear, not what you recognise.</p>
</footer>
<script>
let DATA=null, qi=0, cur=0, ratings={}, audio=new Audio();
const $=id=>document.getElementById(id);

async function boot(){
  DATA=await (await fetch('/data')).json();
  ratings=DATA.ratings||{};
  const first=DATA.queries.findIndex((q,i)=>items(i).some(it=>!(key(i,it) in ratings)));
  qi=first<0?0:first;
  render();
}
const items=i=>DATA.items.filter(x=>x.query_id===i).sort((a,b)=>a.position-b.position);
const key=(i,it)=>i+':'+it.alias;

function render(){
  const q=DATA.queries[qi], its=items(qi);
  $('query').textContent=q.query;
  const rated=Object.keys(ratings).length;
  $('meta').textContent=`${q.category} · query ${qi+1} of ${DATA.queries.length} · ${rated} of ${DATA.items.length} clips rated`;
  $('prog').style.width=(100*rated/DATA.items.length)+'%';
  $('items').innerHTML=its.map((it,n)=>`
    <div class="item ${n===cur?'current':''}" id="it${n}">
      <div class="row">
        <button class="play" onclick="play(${n})">▶ Clip ${n+1}</button>
        <span class="spacer"></span>
        <span class="rate">${[1,2,3,4,5].map(v=>
          `<button class="${ratings[key(qi,it)]===v?'sel':''}" onclick="rate(${n},${v})">${v}</button>`).join('')}</span>
      </div>
    </div>`).join('');
}
function play(n){
  const it=items(qi)[n]; cur=n; render();
  audio.pause(); audio=new Audio('/audio/'+encodeURIComponent(it.audio));
  // Start ~30% in: intros are the least representative part of a track.
  audio.addEventListener('loadedmetadata',()=>{ audio.currentTime=audio.duration*0.3; audio.play(); });
}
function rate(n,v){
  const it=items(qi)[n]; ratings[key(qi,it)]=v; render(); save(false);
  if(n<items(qi).length-1){ cur=n+1; render(); }
  else if(items(qi).every(x=>key(qi,x) in ratings)){ setTimeout(()=>go(1),250); }
}
function go(d){ qi=Math.min(Math.max(qi+d,0),DATA.queries.length-1); cur=0; audio.pause(); render(); }
async function save(explicit){
  await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ratings})});
  if(explicit) $('meta').textContent+=' · saved';
}
document.addEventListener('keydown',e=>{
  if(e.key===' '){e.preventDefault(); audio.paused?audio.play():audio.pause();}
  else if(e.key>='1'&&e.key<='5') rate(cur,+e.key);
  else if(e.key==='j'||e.key==='ArrowDown'){cur=Math.min(cur+1,items(qi).length-1);render();}
  else if(e.key==='k'||e.key==='ArrowUp'){cur=Math.max(cur-1,0);render();}
  else if(e.key==='n') go(1);
  else if(e.key==='p') go(-1);
});
boot();
</script>
"""


class Handler(SimpleHTTPRequestHandler):
    rating_set: dict = {}

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import unquote

        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path == "/data":
            payload = dict(self.rating_set)
            payload["ratings"] = json.loads(RATINGS.read_text())["ratings"] if RATINGS.exists() else {}
            return self._send(200, json.dumps(payload).encode(), "application/json")
        if self.path.startswith("/audio/"):
            rel = unquote(self.path[len("/audio/"):])
            target = (ROOT / rel).resolve()
            # Never serve outside the project: the path comes from the page, and the page
            # is the one thing here that a stray edit could point anywhere.
            if not str(target).startswith(str(ROOT)) or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "none")
            self.end_headers()
            self.wfile.write(data)
            return
        return self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/save":
            return self._send(404, b"not found", "text/plain")
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        ratings = json.loads(body).get("ratings", {})
        RATINGS.write_text(json.dumps({
            "rating_set": "evals/rating_set.json",
            "encoder": self.rating_set.get("encoder"),
            "n_segments": self.rating_set.get("n_segments"),
            "ratings": ratings,
        }, indent=2))
        return self._send(200, b'{"ok":true}', "application/json")

    def log_message(self, *args) -> None:  # keep the console quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Blind relevance rating server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    path = EVALS / "rating_set.private.json"
    if not path.exists():
        print("missing evals/rating_set.private.json — run scripts/build_rating_set.py first",
              file=sys.stderr)
        return 1
    Handler.rating_set = json.loads(path.read_text())

    n_done = len(json.loads(RATINGS.read_text())["ratings"]) if RATINGS.exists() else 0
    url = f"http://127.0.0.1:{args.port}/"
    print(f"rating {len(Handler.rating_set['items'])} clips across "
          f"{len(Handler.rating_set['queries'])} queries; {n_done} already rated")
    print(f"open {url}   (ratings save automatically to {RATINGS.relative_to(ROOT)})")
    if not args.no_open:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
