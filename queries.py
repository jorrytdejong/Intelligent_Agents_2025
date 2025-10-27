from __future__ import annotations

# ============================== Imports ==============================
from owlready2 import get_ontology, default_world
from typing_extensions import TypedDict
from pydantic import BaseModel
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional, Iterable
from collections import defaultdict, deque
from datetime import datetime
import argparse
import sys
import re
import csv
import json
import os
import html
import ollama


# ============================ Data Structures ============================
class InputState(TypedDict):
    story: str

class InbetweenState(TypedDict):
    story: str
    chunks: List[str]
    ontology_summary: str
    extract: Dict[str, List[Dict[str, str]]]
    conflicts: Dict[str, List[Dict[str, str]]]
    rewritten_chunks: List[str]
    re_verified_chunks: List[str]
    evaluated_chunks: List[str]

class OutputState(TypedDict):
    revised_story: str

class Triple(BaseModel):
    subj: str
    pred: str
    obj: str

@dataclass(frozen=True)
class Metrics:
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    #-----------------------ONTOLOGY
ONTOLOGY_PATH = "/Users/jorrytdejong/Library/Mobile Documents/com~apple~CloudDocs/Master Artificial Intelligence/Intelligent Agents/final_handing_in/Intelligent_Agents_2025/IA_ontology4.owl"  # <- adjust if needed
onto = get_ontology(ONTOLOGY_PATH).load()

EX = getattr(onto, "base_iri", "http://IA.org/onto.owl#").rstrip("#") + "#"
PREFIX_EX = f"PREFIX ex: <{EX}>"
PREFIX_XSD = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>"

print("=== Ontology Loaded ===")
print("Base IRI:", EX)
print("Classes:", [c.name for c in onto.classes()])
print("Individuals:", [i.name for i in onto.individuals()])
print("Object Properties:", [p.name for p in onto.object_properties()])
print("Data Properties:", [p.name for p in onto.data_properties()])

# ============================== Utilities ==============================
def _canon(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def _strip_iri(x: str) -> str:
    if not x:
        return x
    return str(x).split(".")[-1].split("/")[-1].split("#")[-1]

def _norm_obj(x: str) -> str:
    return _canon(_strip_iri(x))

def _norm_pred(x: str) -> str:
    x = _canon(x)
    return {
        "islocatedin": "islocatedin",
        "hasage": "hasage",
        "hasoccupation": "hasoccupation",
        "hastrait": "hastrait",
        "haspopulation": "haspopulation",
        "description": "description",
        "requiresresource": "requiresresource",
        "requirestool": "requirestool",
        "ismarriedto": "ismarriedto",
        "hasresourcestatus": "hasresourcestatus",
    }.get(x, x)

def _to_int_maybe(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        num = ""
        for ch in str(x):
            if ch.isdigit():
                num += ch
            elif num:
                break
        return int(num) if num else None

def _safe_ratio(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else 0.0

def _label_of(gold: int, pred: int) -> str:
    if gold == 1 and pred == 1: return "TP"
    if gold == 0 and pred == 0: return "TN"
    if gold == 0 and pred == 1: return "FP"
    return "FN"

# ============================== Step 1: Chunk ==============================
def chunk(state: InputState) -> InbetweenState:
    sentences = re.split(r'(?<=[.!?])\s+', state['story'].strip())
    chunks = [s.strip() for s in sentences if s.strip()]
    print("\n=== Node: Chunk ===")
    print(chunks)
    return {
        "story": state['story'],
        "chunks": chunks,
        "ontology_summary": "",
        "extract": {},
        "conflicts": {},
        "rewritten_chunks": [],
        "re_verified_chunks": [],
        "evaluated_chunks": []
    }

# ============================== Step 2: Extract ==============================
def extract(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Extract Triples ===")
    extracts: Dict[str, List[Dict[str, str]]] = {}
    previous_subject: Optional[str] = None
    onto_inds = {i.name for i in onto.individuals()}

    def norm_subj(s: str) -> str:
        s = s.strip()
        if s in {"She", "He"} and previous_subject:
            return previous_subject
        s = s.title()
        if s not in onto_inds and s.endswith("s") and s[:-1] in onto_inds:
            s = s[:-1]
        return s

    for i, chunk_text in enumerate(state['chunks']):
        triples: List[Dict[str, str]] = []
        seen_triples = set()

        m_subj = re.search(r'\b([A-Z][a-z]+)\b', chunk_text)
        if m_subj:
            previous_subject = m_subj.group(1)

        # Age
        for m in re.finditer(r'\b(She|He|[A-Z][a-z]+)\s+is\s+(\d{1,3})\s+years?\s+old', chunk_text):
            subj = norm_subj(m.group(1))
            triples.append({"subj": subj, "pred": "hasAge", "obj": m.group(2)})

        # Marriage
        for m in re.finditer(r'\b(She|He|[A-Z][a-z]+)\b.*?married.*?\b([A-Z][a-z]+)\b', chunk_text):
            subj = norm_subj(m.group(1))
            triples.append({"subj": subj, "pred": "isMarriedTo", "obj": norm_subj(m.group(2))})

        # City description
        for m in re.finditer(r'\b([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)\s+(?:is\s+a\s+|has\s+a\s+.*?\s+and\s+is\s+a\s+)?(quiet|empty|calm|busy|noisy|crowded)',
                             chunk_text, flags=re.IGNORECASE):
            city_raw = m.group(1).strip()
            city = city_raw.replace(' ', '_')
            desc = m.group(2).lower()
            t = (city, "description", desc)
            if t not in seen_triples:
                triples.append({"subj": city, "pred": "description", "obj": desc})
                seen_triples.add(t)

        # Occupation
        for m in re.finditer(r'\b([A-Z][a-z]+)\b.*?\b(?:works\s+as|is\s+a|is\s+an|became)\s+([A-Za-z][A-Za-z\s\-]+?)(?:[.,;]|$)',
                             chunk_text, flags=re.IGNORECASE):
            subj = norm_subj(m.group(1))
            occ = m.group(2).strip()
            if re.search(r'\b(married|allergic|allergy|peanut|peanuts|pizza|sandwich|reserved person|quiet|empty|city)\b',
                         occ, flags=re.IGNORECASE):
                continue
            t = (subj, "hasOccupation", occ)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "hasOccupation", "obj": occ})
                seen_triples.add(t)

        # Traits
        trait_pat = (r'(([A-Z][a-z]+)\'s\s+(?:husband|wife|spouse)\s+describes\s+(?:her|him|them)\s+as\s+a?\s+([A-Za-z\s\-]+?)\s+person'
                     r'|\b([A-Z][a-z]+|She|He)\s+is\s+(reserved|social|goodhearted|lazy))')
        for m in re.finditer(trait_pat, chunk_text, flags=re.IGNORECASE):
            if m.group(2):
                subj_raw = m.group(2)
                trait_raw = m.group(3)
            else:
                subj_raw = m.group(4)
                trait_raw = m.group(5)
            subj = norm_subj(subj_raw)
            trait = trait_raw.replace('-', ' ').strip()
            trait = trait.title().replace(' ', '_')
            t = (subj, "hasTrait", trait)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "hasTrait", "obj": trait})
                seen_triples.add(t)

        # Location
        for m in re.finditer(r'\b([A-Z][\w]+(?:\s[A-Z][\w]+)*)\s+is\s+(?:located\s+in|in|at)\s+([A-Z][\w]+(?:\s[A-Z][\w]+)*)\.?',
                             chunk_text, flags=re.IGNORECASE):
            subj_raw = m.group(1).strip()
            obj_raw = m.group(2).strip()
            subj_words = subj_raw.split()
            if subj_words and subj_words[0] in {"The", "A", "An"}:
                subj_raw = " ".join(subj_words[1:])
            if not subj_raw:
                continue
            subj = subj_raw.replace(' ', '_')
            obj = norm_subj(obj_raw).replace(' ', '_')
            t = (subj, "isLocatedIn", obj)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "isLocatedIn", "obj": obj})
                seen_triples.add(t)

        # No electricity
        for m in re.finditer(r'\b(?:no|without|lack\s+of)\s+([A-Za-z]+)\s+in\s+([A-Z][a-z]+)\b',
                             chunk_text, flags=re.IGNORECASE):
            resource = m.group(1).title()
            location = m.group(2).title().replace(' ', '_')
            status = 'NoElectricity' if resource == 'Electricity' else f'No{resource}'
            t = (location, "hasResourceStatus", status)
            if t not in seen_triples:
                triples.append({"subj": location, "pred": "hasResourceStatus", "obj": status})
                seen_triples.add(t)

        # Population
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(?:a\s+)?(high|low|large|small)\s+population\b',
                             chunk_text, flags=re.IGNORECASE):
            subj = m.group(1).replace(' ', '_')
            pop = m.group(2).lower()
            t = (subj, "hasPopulation", pop)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "hasPopulation", "obj": pop})
                seen_triples.add(t)

        # Tools
        for m in re.finditer(r'\b(croissants|baguettes|cake|macaron|food)\s+.*?\b(?:made|prepared)\s+in\s+the\s+(oven|stove|pan)\b',
                             chunk_text, flags=re.IGNORECASE):
            tool = m.group(2).title()
            activity = {'Oven': 'Baking', 'Stove': 'Cooking'}.get(tool, 'Activity')
            t = (activity, "requiresTool", tool)
            if t not in seen_triples:
                triples.append({"subj": activity, "pred": "requiresTool", "obj": tool})
                seen_triples.add(t)

        extracts[f"chunk_{i}"] = triples
        print(f"chunk_{i} -> {triples}")

    state['extract'] = extracts
    return state

# ======================= Step 3: Check Conflicts (SPARQL+rules) =======================
_EIFFEL_ALIASES = {"eiffel_tower", "eiffel tower", "eiffel"}

def _sparql(q: str):
    # Owlready2 default_world.sparql returns a generator-like; cast to list safely
    return list(default_world.sparql(q, error_on_undefined_entities=False))

def check_conflicts(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Check Conflicts with SPARQL ===")
    conflicts: Dict[str, List[Dict[str, str]]] = {}

    for chunk_name, triples in state['extract'].items():
        chunk_conflicts: List[Dict[str, str]] = []

        for t in triples:
            subj = _strip_iri(t['subj']).replace(' ', '_')
            pred = _norm_pred(t['pred'])
            obj  = _strip_iri(t['obj']).replace(' ', '_')

            # --- Rule-style checks (deterministic, fast) ---
            # Age < 18
            if pred == "hasage":
                age = _to_int_maybe(obj)
                if age is None or age < 18:
                    chunk_conflicts.append(t)
                    continue

            # Eiffel Tower location
            if pred == "islocatedin" and _canon(subj) in _EIFFEL_ALIASES:
                if _canon(obj) != "paris":
                    chunk_conflicts.append(t)
                    continue

            # City quiet vs high/large pop: cross-validation happens in evaluation; here, try ontology class check if available
            if pred == "description" and obj in {"quiet", "empty", "calm"}:
                q = f"""
{PREFIX_EX}
SELECT ?city WHERE {{
  BIND(ex:{subj} AS ?city) .
  ?city a ?cls .
  FILTER(STRAFTER(STR(?cls), "#") IN ("BusyCity","CrowdedCity","NoisyCity"))
}}
"""
                try:
                    if _sparql(q):
                        chunk_conflicts.append(t)
                        continue
                except Exception as e:
                    print(f"[SPARQL warn] BusyCity check failed for {subj}: {e}")

            # --- Ontology lookups bound to the actual subject ---

            if pred == "islocatedin":
                q = f"""
{PREFIX_EX}
SELECT ?loc WHERE {{
  BIND(ex:{subj} AS ?ind) .
  ?ind ex:isLocatedIn ?loc .
}}
"""
                try:
                    results = _sparql(q)
                    if not results:
                        # unknown in ontology → flag as conflict candidate
                        chunk_conflicts.append(t)
                    else:
                        onto_locs = {_strip_iri(r[0]) for r in results}
                        if obj not in onto_locs:
                            chunk_conflicts.append(t)
                except Exception as e:
                    print(f"[SPARQL warn] isLocatedIn check failed for {subj}: {e}")
                    chunk_conflicts.append(t)
                continue  # avoid running a second generic query below

            if pred == "hastrait":
                q = f"""
{PREFIX_EX}
SELECT ?trait WHERE {{
  BIND(ex:{subj} AS ?ind) .
  ?ind ex:hasTrait ?trait .
}}
"""
                try:
                    results = _sparql(q)
                    if results:
                        onto_traits = {_strip_iri(r[0]) for r in results}
                        if obj not in onto_traits:
                            chunk_conflicts.append(t)
                    else:
                        chunk_conflicts.append(t)
                except Exception as e:
                    print(f"[SPARQL warn] hasTrait check failed for {subj}: {e}")
                    chunk_conflicts.append(t)
                continue

            if pred == "hasresourcestatus" and obj == "NoElectricity":
                # If ontology states some method requiring electricity while city has NoElectricity, it's a potential conflict –
                # but the *pairing* is evaluated later. Here, flag presence to drive rewriting.
                # Optionally verify Oven requires Electricity:
                q = f"""
{PREFIX_EX}
SELECT ?res WHERE {{
  ex:Oven ex:requiresResource ?res .
}}
"""
                try:
                    results = _sparql(q)
                    if results:
                        required = {_strip_iri(r[0]).lower() for r in results}
                        if "electricity" in required:
                            chunk_conflicts.append(t)
                    else:
                        chunk_conflicts.append(t)
                except Exception as e:
                    print(f"[SPARQL warn] requiresResource(Oven) failed: {e}")
                    chunk_conflicts.append(t)
                continue

            if pred == "requirestool":
                # Check method-tool pairing exists
                q = f"""
{PREFIX_EX}
SELECT ?tool WHERE {{
  ex:{subj} a ex:CookingMethod ;
            ex:requiresTool ?tool .
}}
"""
                try:
                    results = _sparql(q)
                    if results:
                        tools = {_strip_iri(r[0]) for r in results}
                        if obj not in tools:
                            chunk_conflicts.append(t)
                    else:
                        chunk_conflicts.append(t)
                except Exception as e:
                    print(f"[SPARQL warn] requiresTool check failed for {subj}: {e}")
                    chunk_conflicts.append(t)
                continue

            if pred == "hasoccupation":
                # If ontology constrains single occupation, check if >1 exists for this person.
                q = f"""
{PREFIX_EX}
SELECT (COUNT(DISTINCT ?occ) AS ?n) WHERE {{
  ex:{subj} ex:hasOccupation ?occ .
}}
"""
                try:
                    results = _sparql(q)
                    if results:
                        # results[0][0] is an rdflib Literal; compare as int if possible
                        n_txt = str(results[0][0])
                        n_val = _to_int_maybe(n_txt)
                        if n_val and n_val > 1:
                            chunk_conflicts.append(t)
                    else:
                        # unknown person/occupation → cannot verify → treat as conflict to be conservative
                        chunk_conflicts.append(t)
                except Exception as e:
                    print(f"[SPARQL warn] hasOccupation count failed for {subj}: {e}")
                    chunk_conflicts.append(t)
                continue

            # Generic fallback if nothing matched above (rare)
            # For some earlier branches, we already continued.
            # If we get here, we do not have a specific check; don't flag by default.

        conflicts[chunk_name] = chunk_conflicts
        print(f"{chunk_name} conflicts -> {chunk_conflicts}")

    state['conflicts'] = conflicts
    return state

# ============================== Step 4: Rewrite ==============================
def rewrite_chunk(chunk_text: str, conflicts_for_chunk: List[Dict[str, str]]) -> str:
    if not conflicts_for_chunk:
        return chunk_text

    summary = "\n".join([f"{c['subj']} {c['pred']} {c['obj']}" for c in conflicts_for_chunk])
    prompt = f"""
You are a highly constrained, specialized text editor. Fix ONLY the inconsistent facts in the story chunk based on the provided triples. Return ONLY the rewritten chunk.

Story chunk:
{chunk_text}

Detected inconsistencies (triples):
{summary}

Rules:
- For hasAge: if < 18, increase minimally to >= 18; if > 110, decrease toward a plausible value (<= 110).
- For isLocatedIn: move subjects like Eiffel_Tower to the ontology-consistent location (e.g., Paris).
- For hasOccupation where there are too many: drop extra occupations to keep within limits.
- For NoElectricity together with oven/baking: adjust so they are not incompatible (e.g., remove the oven use or restore electricity).
- Change ONLY what is needed; preserve consistent content; no explanations.
"""
    resp = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
    # Ollama returns a dict: {"message":{"role":"assistant","content":"..."}}
    try:
        return resp["message"]["content"].strip()
    except Exception:
        # Fallback if a different shape is returned
        return str(resp).strip()

def rewrite_conflicts(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Rewrite Conflicts ===")
    rewritten: List[str] = []
    for i, chunk_text in enumerate(state['chunks']):
        name = f"chunk_{i}"
        rewritten_chunk = rewrite_chunk(chunk_text, state['conflicts'].get(name, []))
        rewritten.append(rewritten_chunk)
        print(f"{name}: {rewritten_chunk}")
    state['rewritten_chunks'] = rewritten
    return state

# ============================== Step 5: Re-verify ==============================
def reverify_chunks(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Re-verify ===")
    temp_state: InbetweenState = {
        "story": " ".join(state['rewritten_chunks']),
        "chunks": state['rewritten_chunks'],
        "ontology_summary": "",
        "extract": {},
        "conflicts": {},
        "rewritten_chunks": [],
        "re_verified_chunks": [],
        "evaluated_chunks": []
    }
    temp_state = extract(temp_state)
    temp_state = check_conflicts(temp_state)

    # keep verified chunks; also keep re-computed extract/conflicts if you want to inspect
    state['re_verified_chunks'] = state['rewritten_chunks']
    # Optionally store for later evaluation
    # state['extract_after'] = temp_state['extract']
    # state['conflicts_after'] = temp_state['conflicts']
    return state

    # ============================== Step 6: Assemble ==============================
def assemble_final_story(state: InbetweenState) -> OutputState:
    final_story = " ".join(state['re_verified_chunks'])
    print("\n=== Node: Final Story ===")
    print(final_story)
    return {"revised_story": final_story}

    # ============================== Step 7: Eval ==============================
def flatten_extractions(extract_dict: Dict[str, List[Dict[str, str]]]) -> List[Tuple[str, Dict[str, str]]]:
    items: List[Tuple[str, Dict[str, str]]] = []
    for chunk_name, triples in (extract_dict or {}).items():
        for t in (triples or []):
            items.append((chunk_name, {"subj": str(t.get("subj","")),
                                      "pred": str(t.get("pred","")),
                                      "obj":  str(t.get("obj",""))}))
    return items

_PRONOUNS = {"she","he","her","him","they","them","his","their"}
_BAD_SUBJECTS = {"and","the","a","an","city"}

def _resolve_subjects_in_story(chunks: List[str]) -> Dict[str, str]:
    last_name = ""
    proper = re.compile(r"^[A-Z][a-z]+$")
    window = deque(maxlen=3)
    for ch in chunks or []:
        toks = re.findall(r"[A-Za-z]+", ch)
        for t in toks:
            if proper.match(t) and t.lower() != "i":
                last_name = t
                window.append(t)
    return {"_last": last_name or (window[-1] if window else "")}

def _clean_subject(subj: str, coref: Dict[str, str]) -> str:
    s = _canon(subj)
    if s in _PRONOUNS and coref.get("_last"):
        return _canon(coref["_last"])
    if s in _BAD_SUBJECTS:
        return ""
    return s

def _is_conflict_evaluable(t: Dict[str, str]) -> bool:
    pred = _norm_pred(t.get("pred",""))
    subj = _canon(t.get("subj",""))
    obj  = _norm_obj(t.get("obj",""))
    if not pred or not obj or not subj:
        return False
    if subj in _BAD_SUBJECTS:
        return False
    if pred == "haspopulation" and obj in {"high","large"}:
        return False
    return True

def violation_oracle(triple: Dict[str, str]) -> int:
    subj = _canon(triple.get("subj",""))
    pred = _norm_pred(triple.get("pred",""))
    obj  = _norm_obj(triple.get("obj",""))

    if pred == "hasage":
        age_val = _to_int_maybe(triple.get("obj",""))
        return 1 if (age_val is None or age_val < 18) else 0
    if pred == "islocatedin" and subj in _EIFFEL_ALIASES:
        return 1 if obj != "paris" else 0
    return 0

def collect_labels(state: InbetweenState) -> Tuple[List[int], List[int], List[Tuple[str, Dict[str, str]]]]:
    flat = flatten_extractions(state.get("extract", {}))
    coref_hint = _resolve_subjects_in_story(state.get("chunks", []))

    conflict_set = set()
    for ch, triples in (state.get("conflicts", {}) or {}).items():
        for t in (triples or []):
            if not _is_conflict_evaluable(t):
                continue
            key = (
                _canon(ch),
                _clean_subject(t.get("subj",""), coref_hint),
                _norm_pred(t.get("pred","")),
                _norm_obj(t.get("obj","")),
            )
            if key[1] and key[2]:
                conflict_set.add(key)

    occ_seen: Dict[str, int] = defaultdict(int)
    city_pop: Dict[str, bool] = defaultdict(bool)
    city_quiet: Dict[str, bool] = defaultdict(bool)
    has_no_elec = False
    used_oven   = False
    jan_reserved = False
    jan_social   = False

    normalized_flat: List[Tuple[str, Dict[str,str]]] = []
    for chunk_name, t in flat:
        subj = _clean_subject(t.get("subj",""), coref_hint)
        pred = _norm_pred(t.get("pred",""))
        obj  = _norm_obj(t.get("obj",""))
        normalized_flat.append((chunk_name, {"subj": subj, "pred": pred, "obj": obj}))

        if pred == "haspopulation" and obj in {"high","large"}:
            city_pop[subj] = True
        if pred == "description" and any(k in obj for k in ["quiet","empty","calm"]):
            city_quiet[subj] = True
        if pred == "hastrait" and subj == "jan":
            if obj == "reserved": jan_reserved = True
            if obj == "social":   jan_social = True
        if pred == "requirestool" and obj == "oven":
            used_oven = True
        if pred == "hasresourcestatus" and obj == "noelectricity":
            has_no_elec = True

    y_true: List[int] = []
    y_pred: List[int] = []
    aligned: List[Tuple[str, Dict[str, str]]] = []

    for chunk_name, t in normalized_flat:
        subj, pred, obj = t["subj"], t["pred"], t["obj"]
        gold = violation_oracle(t)

        if pred == "hasoccupation" and subj:
            occ_seen[subj] += 1
            if occ_seen[subj] > 1:
                gold = 1
        if pred in {"haspopulation","description"} and subj and city_pop.get(subj) and city_quiet.get(subj):
            gold = 1
        if subj == "jan" and pred == "hastrait" and (jan_reserved and jan_social):
            gold = 1
        if used_oven and has_no_elec and pred in {"hasresourcestatus","requirestool"}:
            gold = 1

        pred_label = 1 if ((_canon(chunk_name), subj, pred, obj) in conflict_set) else 0

        y_true.append(gold)
        y_pred.append(pred_label)
        aligned.append((chunk_name, {"subj": subj, "pred": pred, "obj": obj}))

    return y_true, y_pred, aligned

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return Metrics(tp=0, tn=0, fp=0, fn=0, precision=0.0, recall=0.0, f1=0.0, accuracy=0.0).__dict__
    yt, yp = y_true[:n], y_pred[:n]
    tp = sum(1 for a,b in zip(yt, yp) if a==1 and b==1)
    tn = sum(1 for a,b in zip(yt, yp) if a==0 and b==0)
    fp = sum(1 for a,b in zip(yt, yp) if a==0 and b==1)
    fn = sum(1 for a,b in zip(yt, yp) if a==1 and b==0)
    precision = _safe_ratio(tp, tp+fp)
    recall    = _safe_ratio(tp, tp+fn)
    f1        = _safe_ratio(2*precision*recall, precision+recall) if (precision or recall) else 0.0
    accuracy  = _safe_ratio(tp+tn, tp+tn+fp+fn)
    return Metrics(tp=tp, tn=tn, fp=fp, fn=fn,
                   precision=precision, recall=recall, f1=f1, accuracy=accuracy).__dict__

def reputation_scores(aligned_records: List[Tuple[str, Dict[str, str]]], y_true: List[int]) -> Dict[str, Any]:
    total = len(y_true)
    overall = _safe_ratio(sum(1 for v in y_true if v == 0), total) if total else 1.0
    by_chunk_gold: Dict[str, List[int]] = defaultdict(list)
    by_subject_gold: Dict[str, List[int]] = defaultdict(list)
    for (chunk_name, t), gold in zip(aligned_records, y_true):
        by_chunk_gold[chunk_name].append(gold)
        by_subject_gold[t.get("subj","")].append(gold)
    chunk_rep = {ch: _safe_ratio(vals.count(0), len(vals)) for ch, vals in by_chunk_gold.items() if vals}
    subj_rep  = {sj: _safe_ratio(vals.count(0), len(vals)) for sj, vals in by_subject_gold.items() if vals}
    return {"overall": overall, "by_chunk": chunk_rep, "by_subject": subj_rep}

def recompute_on_text(chunks: List[str]) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    temp_state: InbetweenState = {
        "story": " ".join(chunks or []),
        "chunks": chunks or [],
        "ontology_summary": "",
        "extract": {},
        "conflicts": {},
        "rewritten_chunks": [],
        "re_verified_chunks": [],
        "evaluated_chunks": []
    }
    temp_state = extract(temp_state)
    temp_state = check_conflicts(temp_state)
    return temp_state.get("extract", {}), temp_state.get("conflicts", {})

def rows_for_export(aligned: List[Tuple[str, Dict[str, str]]],
                    y_true: List[int], y_pred: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n = min(len(aligned), len(y_true), len(y_pred))
    for i in range(n):
        chunk_name, t = aligned[i]
        gold, pred = y_true[i], y_pred[i]
        rows.append({
            "chunk": chunk_name,
            "subject": t.get("subj",""),
            "predicate": t.get("pred",""),
            "object": t.get("obj",""),
            "gold_violation": gold,
            "pred_violation": pred,
            "label": _label_of(gold, pred)
        })
    return rows

def run_one_story(story_text: str) -> Dict[str, Any]:
    state = chunk({"story": story_text})
    state = extract(state)
    state = check_conflicts(state)

    y_true_b, y_pred_b, aligned_b = collect_labels(state)
    mb = compute_metrics(y_true_b, y_pred_b)
    vb = sum(y_true_b)

    state = rewrite_conflicts(state)
    state = reverify_chunks(state)

    ex_after, conf_after = recompute_on_text(state.get("re_verified_chunks", []))
    temp_after: InbetweenState = {
        "story": " ".join(state.get("re_verified_chunks", [])),
        "chunks": state.get("re_verified_chunks", []),
        "ontology_summary": "",
        "extract": ex_after,
        "conflicts": conf_after,
        "rewritten_chunks": [],
        "re_verified_chunks": [],
        "evaluated_chunks": []
    }
    y_true_a, y_pred_a, aligned_a = collect_labels(temp_after)
    ma = compute_metrics(y_true_a, y_pred_a)
    va = sum(y_true_a)

    return {"before": mb, "after": ma, "viol_before": vb, "viol_after": va}

def micro_sum(metrics_list: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(m.get("tp", 0)) for m in metrics_list)
    fp = sum(int(m.get("fp", 0)) for m in metrics_list)
    fn = sum(int(m.get("fn", 0)) for m in metrics_list)
    tn = sum(int(m.get("tn", 0)) for m in metrics_list)
    precision = _safe_ratio(tp, tp+fp)
    recall    = _safe_ratio(tp, tp+fn)
    f1        = _safe_ratio(2*precision*recall, precision+recall) if (precision or recall) else 0.0
    accuracy  = _safe_ratio(tp+tn, tp+tn+fp+fn)
    return Metrics(tp=tp, fp=fp, fn=fn, tn=tn,
                   precision=precision, recall=recall, f1=f1, accuracy=accuracy).__dict__

def run_suite(stories: List[str]):
    runs = [run_one_story(s) for s in (stories or [])]
    micro_before = micro_sum((r["before"] for r in runs))
    micro_after  = micro_sum((r["after"]  for r in runs))
    viol_before  = sum(int(r["viol_before"]) for r in runs) or 0
    viol_after   = sum(int(r["viol_after"])  for r in runs) or 0
    resolution   = _safe_ratio((viol_before - viol_after), viol_before) if viol_before else 0.0
    print("\n=== MICRO-AVERAGED (across stories) ===")
    print("Before:", micro_before)
    print("After:", micro_after)
    print(f"Violations (gold): {viol_before} → {viol_after}  (Resolution: {resolution:.3f})")
    return runs, micro_before, micro_after, resolution

def _print_reputation(title: str, rep: Dict[str, Any], top_k: int = 5) -> None:
    """
    Pretty-print reputation scores.
    rep = {"overall": float, "by_chunk": {chunk: score}, "by_subject": {subject: score}}
    Higher is better (1.0 means no gold violations).
    """
    print(f"\n=== REPUTATION — {title} ===")
    print(f"Overall: {rep.get('overall', 0.0):.3f}")

    by_chunk = rep.get("by_chunk", {}) or {}
    by_subject = rep.get("by_subject", {}) or {}

    if by_chunk:
        worst_chunks = sorted(by_chunk.items(), key=lambda kv: kv[1])[:top_k]
        best_chunks  = sorted(by_chunk.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        print("  Worst chunks (↓): " + ", ".join([f"{k}:{v:.2f}" for k, v in worst_chunks]))
        print("  Best  chunks (↑): "  + ", ".join([f"{k}:{v:.2f}" for k, v in best_chunks]))
    else:
        print("  (no chunk-level records)")

    if by_subject:
        worst_subj = sorted(by_subject.items(), key=lambda kv: kv[1])[:top_k]
        best_subj  = sorted(by_subject.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        print("  Worst subjects (↓): " + ", ".join([f"{(k or '<empty>')}:{v:.2f}" for k, v in worst_subj]))
        print("  Best  subjects (↑): "  + ", ".join([f"{(k or '<empty>')}:{v:.2f}" for k, v in best_subj]))
    else:
        print("  (no subject-level records)")

# ============================== Main / Scenarios ==============================
if __name__ == "__main__":
    scenarios = {
        1: """Amelia is one of France's most famous lawyers. She is 17 years old and recently married her husband Adam, who is 25.""",
        2: """They live together in Paris, a quiet city without much hustle and bustle since the city has a large population.
The city is located on flat land, so there are no mountains.""",
        3: """The Eiffel Tower is located in Italy.""",
        4: """Her husband Adam is a baker and a merchant. He makes various types of French pastries, including macarons, but also baguettes and croissants.
Unfortunately, Amelia cannot eat many sweets because she is allergic to the filling which contains dairy.""",
        5: """Amelia's husband describes her as a good hearted and self serving person who cares about the well-being of people.""",
        6: """It was the day that there was no electricity in Paris. Amelia went to the bakery where Adam worked.
When Amelia came in, she immediately bought two croissants which were made in the oven.""",
        7: """Los Angeles has a high population and is a quiet and empty city""",
        8: """Jan is 115 years old.""",
        9: """Jan is reserved. During lunch Jan enjoys conversations with other patients, nurses, and all the other staff."""
    }

    import argparse, sys
    ap = argparse.ArgumentParser(description="Run full story pipeline on scenarios or custom text.")
    ap.add_argument("--scenario", type=int, default=None, help="Scenario number (1-9). Ignored if --story or --all.")
    ap.add_argument("--story", type=str, default=None, help="Custom story text to process (single run).")
    ap.add_argument("--all", action="store_true", help="Run ALL predefined scenarios (1-9).")
    args, _unknown = ap.parse_known_args()

    def run_full_once(story_text: str, label: str):
        print(f"\n\n#############################")
        print(f"### RUNNING {label.upper()} ###")
        print(f"#############################\n{story_text}\n")

        # --- BEFORE ---
        state = chunk({"story": story_text})
        state = extract(state)
        state = check_conflicts(state)

        y_true_b, y_pred_b, aligned_b = collect_labels(state)
        mb = compute_metrics(y_true_b, y_pred_b)
        vb = sum(y_true_b)

        print("\n=== METRICS — BEFORE REWRITE ===")
        print(mb)
        print(f"Gold violations (sum of y_true): {vb}")

        rep_b = reputation_scores(aligned_b, y_true_b)
        _print_reputation("BEFORE", rep_b, top_k=5)

        # --- REWRITE + RE-VERIFY ---
        state = rewrite_conflicts(state)
        state = reverify_chunks(state)

        ex_after, conf_after = recompute_on_text(state.get("re_verified_chunks", []))
        temp_after: InbetweenState = {
            "story": " ".join(state.get("re_verified_chunks", [])),
            "chunks": state.get("re_verified_chunks", []),
            "ontology_summary": "",
            "extract": ex_after,
            "conflicts": conf_after,
            "rewritten_chunks": [],
            "re_verified_chunks": [],
            "evaluated_chunks": []
        }

        # --- AFTER ---
        y_true_a, y_pred_a, aligned_a = collect_labels(temp_after)
        ma = compute_metrics(y_true_a, y_pred_a)
        va = sum(y_true_a)

        print("\n=== METRICS — AFTER REWRITE ===")
        print(ma)
        print(f"Gold violations (sum of y_true): {va}")

        rep_a = reputation_scores(aligned_a, y_true_a)
        _print_reputation("AFTER", rep_a, top_k=5)

        resolution = ((vb - va) / vb) if vb else 0.0
        print("\n=== SUMMARY ===")
        print(f"Resolution: {resolution:.3f}  (gold violations {vb} → {va})")

        # --- Final story ---
        try:
            final_state = assemble_final_story(state)
            revised = final_state.get("revised_story", "")
        except NameError:
            revised = " ".join(state.get("re_verified_chunks") or state.get("rewritten_chunks") or state.get("chunks") or [])

        print("\n=== FINAL OUTPUT ===")
        print(revised if revised else "[main] No revised story produced.")

        # Return for micro-averaging
        return {
            "before": mb,
            "after": ma,
            "viol_before": vb,
            "viol_after": va
        }

    # Decide run mode
    runs_meta = []
    if args.story and args.story.strip():
        runs_meta.append(run_full_once(args.story.strip(), "custom"))
    elif args.all or args.scenario is None:
        # Default to ALL if no specific scenario or custom story is provided
        for k in sorted(scenarios.keys()):
            runs_meta.append(run_full_once(scenarios[k], f"scenario {k}"))
    else:
        if args.scenario not in scenarios:
            print(f"[main] Scenario {args.scenario} not found. Choose 1–9, pass --all, or use --story.")
            sys.exit(2)
        runs_meta.append(run_full_once(scenarios[args.scenario], f"scenario {args.scenario}"))

    # === MICRO-AVERAGED OVER ALL RUNS ===
    micro_before = micro_sum([r["before"] for r in runs_meta])
    micro_after  = micro_sum([r["after"]  for r in runs_meta])
    vb_total = sum(int(r["viol_before"]) for r in runs_meta)
    va_total = sum(int(r["viol_after"])  for r in runs_meta)
    resolution_total = ((vb_total - va_total) / vb_total) if vb_total else 0.0

    print("\n\n================== OVERALL (ALL RUNS) ==================")
    print("Micro BEFORE:", micro_before)
    print("Micro AFTER :", micro_after)
    print(f"Total gold violations: {vb_total} → {va_total}  (Resolution: {resolution_total:.3f})")
    print("========================================================\n")
