"""Amostragem e carga do conjunto de imagens do benchmark.

Reusa o AnimalDataset real do pipeline (list_tags / load_index) para o
inventário. A leitura do PNG depth usa skimage.io.imread primeiro (idêntico ao
AnimalDataset.load_depth no Pi); se o skimage estiver ausente (ex.: dry-run no
Mac sem a dep), cai para Pillow — mesmos bytes uint16 mm.

image_id é um hash curto de "tag/depth_filename": assim a MESMA imagem tem o
MESMO identificador anônimo entre os benchmarks de seletor/enhancer/preditor
(§7), sem vazar caminhos pessoais.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

# Suíted (classe positiva do seletor) vs. demais (background/parcial/ruido).
SUITED_LABEL = "suited"


def _load_depth_png(path) -> np.ndarray:
    """Lê o PNG depth preservando uint16 mm. skimage (pipeline) primeiro."""
    try:
        import skimage.io as ski_io
        img = ski_io.imread(str(path))
        return np.asarray(img)
    except Exception:
        pass
    try:
        from PIL import Image
        img = Image.open(str(path))
        arr = np.asarray(img)
        return arr
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Não foi possível ler depth PNG {path}: {e}")


def _image_id(tag: str, depth_filename: str) -> str:
    h = hashlib.sha1(f"{tag}/{depth_filename}".encode("utf-8")).hexdigest()
    return "img_" + h[:8]


# --------------------------------------------------------------------------- #
# Inventário via AnimalDataset real
# --------------------------------------------------------------------------- #
def _try_animal_dataset(data_root: str):
    """Tenta reusar o AnimalDataset real do pipeline (list_tags/load_index).

    Retorna (dataset, tags) ou (None, None) se não for importável (ex.: skimage
    ausente no ambiente de dry-run). No Pi (skimage presente) sempre reusa.
    """
    try:
        from mas.utils.animal_dataset import AnimalDataset
        ds = AnimalDataset(data_root)
        return ds, ds.list_tags()
    except Exception:
        return None, None


def _fallback_tags(data_root: str, limit_tags=None) -> list[str]:
    tags_root = Path(data_root) / "animal-tags"
    tags = sorted(
        p.name for p in tags_root.iterdir()
        if p.is_dir() and (p / "simulation_index.json").exists()
    )
    return tags[:limit_tags] if limit_tags else tags


def _load_index(data_root: str, tag: str, dataset) -> list[dict]:
    """load_index real (AnimalDataset) se disponível; senão lê o JSON direto."""
    if dataset is not None:
        return dataset.load_index(tag)
    with open(Path(data_root) / "animal-tags" / tag
              / "simulation_index.json") as f:
        return json.load(f)


def collect_inventory(data_root: str, limit_tags: int | None = None):
    """Retorna (suited_entries, not_suited_entries) lendo os simulation_index.

    Cada entry: dict(tag, depth_filename, label, true_class, image_id).
    Reusa AnimalDataset.list_tags / load_index do pipeline quando disponível.
    """
    dataset, tags = _try_animal_dataset(data_root)
    if dataset is None:
        tags = _fallback_tags(data_root, limit_tags)
    elif limit_tags:
        tags = tags[:limit_tags]

    suited, notsuited = [], []
    for tag in tags:
        try:
            index = _load_index(data_root, tag, dataset)
        except Exception:
            continue
        for fr in index:
            label = fr.get("label")
            depth_filename = fr.get("depth_filename")
            if not depth_filename:
                continue
            entry = {
                "tag": tag,
                "depth_filename": depth_filename,
                "label": label,
                "true_class": (SUITED_LABEL if label == SUITED_LABEL
                               else "not_suited"),
                "image_id": _image_id(tag, depth_filename),
            }
            (suited if label == SUITED_LABEL else notsuited).append(entry)
    return suited, notsuited


def _round_robin_pick(entries: list[dict], quota: int,
                      rng: random.Random) -> list[dict]:
    """Espalha a seleção entre animais (round-robin por tag embaralhada)."""
    if quota <= 0 or not entries:
        return []
    by_tag: dict[str, list[dict]] = {}
    for e in entries:
        by_tag.setdefault(e["tag"], []).append(e)
    for k in by_tag:  # embaralha dentro de cada tag p/ variar posição/frames
        rng.shuffle(by_tag[k])
    tag_order = list(by_tag.keys())
    rng.shuffle(tag_order)

    picked: list[dict] = []
    cursors = {t: 0 for t in tag_order}
    while len(picked) < quota:
        progressed = False
        for t in tag_order:
            if cursors[t] < len(by_tag[t]):
                picked.append(by_tag[t][cursors[t]])
                cursors[t] += 1
                progressed = True
                if len(picked) >= quota:
                    break
        if not progressed:
            break  # esgotou o inventário antes da quota
    return picked


# --------------------------------------------------------------------------- #
# Pools por componente
# --------------------------------------------------------------------------- #
def build_selector_pool(data_root: str, seed: int,
                        per_class: int, limit_tags: int | None = None):
    """Pool balanceado suited/not_suited (carga de bytes incluída)."""
    suited, notsuited = collect_inventory(data_root, limit_tags)
    rng = random.Random(seed)
    su = _round_robin_pick(suited, per_class, rng)
    ns = _round_robin_pick(notsuited, per_class, rng)
    pool = []
    for e in su + ns:
        img = _load_depth_png(f"{data_root}/DEPTH/{e['tag']}/{e['depth_filename']}")
        pool.append({**e, "img": img})
    rng.shuffle(pool)
    stats = {"suited_unique": len(su), "not_suited_unique": len(ns),
             "total_unique": len(pool), "available_suited": len(suited),
             "available_not_suited": len(notsuited)}
    return pool, stats


def build_suited_pool(data_root: str, seed: int, quota: int,
                      limit_tags: int | None = None):
    """Pool só de frames suited (enhancer/preditor), espalhado por animal."""
    suited, _ = collect_inventory(data_root, limit_tags)
    rng = random.Random(seed)
    su = _round_robin_pick(suited, quota, rng)
    pool = []
    for e in su:
        img = _load_depth_png(f"{data_root}/DEPTH/{e['tag']}/{e['depth_filename']}")
        pool.append({**e, "img": img})
    rng.shuffle(pool)
    return pool, {"suited_unique": len(pool), "available_suited": len(suited)}


# --------------------------------------------------------------------------- #
# Ordem de iteração (embaralhada, cíclica, com contagem de reuso)
# --------------------------------------------------------------------------- #
def cyclic_order(pool_size: int, iterations: int, seed: int):
    """Devolve (indices, reuse_counts).

    indices: lista de tamanho `iterations` com índices em [0, pool_size),
             gerados percorrendo uma permutação embaralhada (seed fixa) e
             reiniciando o ciclo quando acaba — nunca medindo 1000x a mesma
             imagem, e sem agrupar classes/animais.
    reuse_counts: quantas vezes cada imagem foi reutilizada.
    """
    rng = random.Random(seed)
    if pool_size == 0:
        return [], []
    order: list[int] = []
    while len(order) < iterations:
        perm = list(range(pool_size))
        rng.shuffle(perm)
        order.extend(perm)
    order = order[:iterations]
    reuse = [0] * pool_size
    for i in order:
        reuse[i] += 1
    return order, reuse
