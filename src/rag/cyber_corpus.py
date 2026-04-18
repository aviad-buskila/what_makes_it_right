"""Load a canonical, structured cybersecurity knowledge corpus.

Sources (all public, authoritative, machine-readable):
  - MITRE ATT&CK Enterprise — STIX 2.x JSON
  - MITRE CWE                — CSV export of all weaknesses
  - NIST SP 800-53 r5        — OSCAL JSON (controls catalog)
  - OWASP Top 10 (2021)      — markdown from the official OWASP repo

Each source is converted into plain-text "documents" with a short header
(ID + title) so the downstream chunker sees coherent, self-contained text.
Files are cached under ``cache_dir`` to keep build reproducible and offline-
friendly after the first run.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "mitre_attack": (
        "https://raw.githubusercontent.com/mitre/cti/master/"
        "enterprise-attack/enterprise-attack.json"
    ),
    "cwe_csv_zip": (
        "https://cwe.mitre.org/data/csv/1000.csv.zip"
    ),
    "nist_800_53": (
        "https://raw.githubusercontent.com/usnistgov/oscal-content/main/"
        "nist.gov/SP800-53/rev5/json/"
        "NIST_SP-800-53_rev5_catalog.json"
    ),
    "owasp_top10_root": (
        "https://raw.githubusercontent.com/OWASP/Top10/master/2021/docs/"
    ),
}

OWASP_TOP10_2021 = [
    ("A01_2021-Broken_Access_Control.md", "A01:2021 — Broken Access Control"),
    ("A02_2021-Cryptographic_Failures.md", "A02:2021 — Cryptographic Failures"),
    ("A03_2021-Injection.md", "A03:2021 — Injection"),
    ("A04_2021-Insecure_Design.md", "A04:2021 — Insecure Design"),
    ("A05_2021-Security_Misconfiguration.md",
     "A05:2021 — Security Misconfiguration"),
    ("A06_2021-Vulnerable_and_Outdated_Components.md",
     "A06:2021 — Vulnerable and Outdated Components"),
    ("A07_2021-Identification_and_Authentication_Failures.md",
     "A07:2021 — Identification and Authentication Failures"),
    ("A08_2021-Software_and_Data_Integrity_Failures.md",
     "A08:2021 — Software and Data Integrity Failures"),
    ("A09_2021-Security_Logging_and_Monitoring_Failures.md",
     "A09:2021 — Security Logging and Monitoring Failures"),
    ("A10_2021-Server_Side_Request_Forgery_(SSRF).md",
     "A10:2021 — Server-Side Request Forgery (SSRF)"),
]


def _fetch(url: str, cache_path: Path, binary: bool = False) -> bytes | str:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        cache_path.write_bytes(data)
    raw = cache_path.read_bytes()
    return raw if binary else raw.decode("utf-8", errors="replace")


def _load_mitre_attack(cache_dir: Path) -> list[dict[str, str]]:
    raw = _fetch(SOURCES["mitre_attack"], cache_dir / "enterprise-attack.json")
    bundle = json.loads(raw)
    objects = bundle.get("objects", [])
    docs: list[dict[str, str]] = []
    for obj in objects:
        otype = obj.get("type")
        if otype not in {"attack-pattern", "course-of-action",
                         "intrusion-set", "malware", "tool"}:
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        name = (obj.get("name") or "").strip()
        desc = (obj.get("description") or "").strip()
        if not name or not desc:
            continue
        ext_id = ""
        for ref in obj.get("external_references", []) or []:
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id", "")
                break
        tactics = ", ".join(
            phase.get("phase_name", "")
            for phase in obj.get("kill_chain_phases", []) or []
        )
        header_bits = [p for p in (ext_id, name, otype, tactics) if p]
        header = " | ".join(header_bits)
        docs.append({
            "id": f"attack_{ext_id or obj.get('id', name)}",
            "text": f"{header}\n\n{desc}",
        })
    return docs


def _load_cwe(cache_dir: Path) -> list[dict[str, str]]:
    archive = cache_dir / "cwe_1000.csv.zip"
    _fetch(SOURCES["cwe_csv_zip"], archive, binary=True)
    docs: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as zf:
        csv_name = next((n for n in zf.namelist() if n.endswith(".csv")), None)
        if csv_name is None:
            return docs
        with zf.open(csv_name) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text)
            for row in reader:
                cwe_id = (row.get("CWE-ID") or "").strip()
                name = (row.get("Name") or "").strip()
                desc = (row.get("Description") or "").strip()
                extended = (row.get("Extended Description") or "").strip()
                consequences = (row.get("Common Consequences") or "").strip()
                mitigations = (row.get("Potential Mitigations") or "").strip()
                if not cwe_id or not name or not desc:
                    continue
                sections = [f"CWE-{cwe_id} | {name}", desc]
                if extended:
                    sections.append(extended)
                if consequences:
                    sections.append(f"Consequences: {consequences}")
                if mitigations:
                    sections.append(f"Mitigations: {mitigations}")
                docs.append({
                    "id": f"cwe_{cwe_id}",
                    "text": "\n\n".join(sections),
                })
    return docs


def _extract_nist_text(parts: list[dict]) -> str:
    out: list[str] = []
    for part in parts or []:
        prose = (part.get("prose") or "").strip()
        if prose:
            out.append(prose)
        sub = part.get("parts")
        if sub:
            out.append(_extract_nist_text(sub))
    return "\n".join(s for s in out if s)


def _load_nist_800_53(cache_dir: Path) -> list[dict[str, str]]:
    raw = _fetch(SOURCES["nist_800_53"], cache_dir / "nist_800_53_r5.json")
    catalog = json.loads(raw).get("catalog", {})
    docs: list[dict[str, str]] = []
    for group in catalog.get("groups", []) or []:
        family = (group.get("title") or "").strip()
        for control in group.get("controls", []) or []:
            cid = (control.get("id") or "").upper()
            title = (control.get("title") or "").strip()
            body = _extract_nist_text(control.get("parts", []))
            if not cid or not body:
                continue
            header = f"NIST SP 800-53 r5 | {family} | {cid} — {title}"
            docs.append({"id": f"nist80053_{cid}",
                         "text": f"{header}\n\n{body}"})
            for sub in control.get("controls", []) or []:
                sid = (sub.get("id") or "").upper()
                stitle = (sub.get("title") or "").strip()
                sbody = _extract_nist_text(sub.get("parts", []))
                if sid and sbody:
                    sheader = f"NIST SP 800-53 r5 | {family} | {sid} — {stitle}"
                    docs.append({"id": f"nist80053_{sid}",
                                 "text": f"{sheader}\n\n{sbody}"})
    return docs


def _load_owasp_top10(cache_dir: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for file_name, title in OWASP_TOP10_2021:
        url = SOURCES["owasp_top10_root"] + file_name
        try:
            raw = _fetch(url, cache_dir / file_name)
        except Exception as exc:
            print(f"OWASP fetch failed for {file_name}: {exc}. Skipping.")
            continue
        text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
        docs.append({"id": f"owasp_{file_name}",
                     "text": f"OWASP Top 10 (2021) | {title}\n\n{text}"})
    return docs


def load_cybersecurity_corpus(
    cache_dir: str | None = "data/cyber_kb",
    include: tuple[str, ...] = ("attack", "cwe", "nist", "owasp"),
    max_corpus_size: int | None = None,
    random_seed: int = 42,
) -> list[dict[str, str]]:
    """Assemble the cybersecurity knowledge corpus.

    ``include`` lets callers ablate which sources contribute, which is useful
    for the paper's retrieval-provenance analysis.
    """
    cache_root = Path(cache_dir or "data/cyber_kb")
    cache_root.mkdir(parents=True, exist_ok=True)

    corpus: list[dict[str, str]] = []
    if "attack" in include:
        attack_docs = _load_mitre_attack(cache_root / "attack")
        print(f"  MITRE ATT&CK: {len(attack_docs)} documents")
        corpus.extend(attack_docs)
    if "cwe" in include:
        cwe_docs = _load_cwe(cache_root / "cwe")
        print(f"  CWE:          {len(cwe_docs)} documents")
        corpus.extend(cwe_docs)
    if "nist" in include:
        nist_docs = _load_nist_800_53(cache_root / "nist")
        print(f"  NIST 800-53:  {len(nist_docs)} documents")
        corpus.extend(nist_docs)
    if "owasp" in include:
        owasp_docs = _load_owasp_top10(cache_root / "owasp")
        print(f"  OWASP Top 10: {len(owasp_docs)} documents")
        corpus.extend(owasp_docs)

    if max_corpus_size is not None and len(corpus) > max_corpus_size:
        import random
        rng = random.Random(random_seed)
        corpus = rng.sample(corpus, max_corpus_size)
    return corpus
