from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, List, Dict, Any
import hashlib
import json
import http.client
import urllib.parse


EmbedFn = Callable[[str], List[float]]

def _toy_embed(text: str, dim: int = 64) -> List[float]:
    """
    簡易 fallback：把字元碼做 rolling-hash 取固定維度。
    只用於 PoC；請在實戰改成真正的 embedding（HF/OSS/雲端均可）。
    """
    v = [0.0] * dim
    for i, ch in enumerate(text):
        v[i % dim] += (ord(ch) % 97) / 97.0
    # L2 normalize
    import math
    norm = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x / norm for x in v]


@dataclass
class QdrantConfig:
    url: str                 # e.g. http://localhost:6333
    collection: str          # e.g. code
    dim: int = 768           # embedding dimension
    distance: str = "Cosine" # "Cosine" | "Dot" | "Euclid"


class QdrantClientWrapper:
    def __init__(self, cfg: QdrantConfig, embed_fn: Optional[EmbedFn] = None):
        self.cfg = cfg
        self.embed = embed_fn or (lambda t: _toy_embed(t, cfg.dim))
        self._ensure_collection()

    # --- HTTP helpers ---
    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        parsed = urllib.parse.urlparse(self.cfg.url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=30)
        try:
            payload = json.dumps(body).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"}
            conn.request(method, f"{parsed.path.rstrip('/')}{path}", body=payload, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 300:
                raise RuntimeError(f"Qdrant {method} {path} failed: {resp.status} {data[:300]!r}")
            return json.loads(data or b"{}")
        finally:
            conn.close()

    # --- Collection management ---
    def _ensure_collection(self) -> None:
        body = {
            "vectors": {"size": self.cfg.dim, "distance": self.cfg.distance}
        }
        # Create if not exists (idempotent behavior in Qdrant 1.7+ with "recreate":false)
        try:
            self._request("PUT", f"/collections/{self.cfg.collection}", body)
        except Exception:
            # If already exists, ignore
            pass

    # --- Upsert points ---
    def upsert_points(self, items: Iterable[dict]) -> dict:
        """
        items: iterable of dict {
          "id": str,
          "text": str,
          "payload": {...},
          "vector": Optional[List[float]]
        }
        """
        points = []
        for it in items:
            text = it["text"]
            vec  = it.get("vector") or self.embed(text)
            pid  = it.get("id") or hashlib.sha1((it["payload"].get("path","") + str(it["payload"].get("start")) + str(it["payload"].get("end")) + it["payload"].get("qualified_name","")).encode("utf-8")).hexdigest()
            points.append({
                "id": pid,
                "vector": vec,
                "payload": it["payload"],
            })
        body = {"points": points}
        return self._request("PUT", f"/collections/{self.cfg.collection}/points?wait=true", body)

    # --- Query by exact payload (no vector) ---
    def fetch_by_payload(self, payload_filter: Dict[str, Any], limit: int = 3) -> dict:
        body = {
            "filter": {"must": [{"key": k, "match": {"value": v}} for k, v in payload_filter.items()]},
            "limit": limit,
            "with_payload": True,
        }
        return self._request("POST", f"/collections/{self.cfg.collection}/points/scroll", body)
