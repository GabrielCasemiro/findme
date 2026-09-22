"""Group face embeddings into unique people — quality-aware.

The naive "cluster every face" approach over-segments a real event: blurry, side-on,
grimacing, or tiny background faces produce noisy embeddings that neither match a
person's good faces (→ hundreds of junk singletons) nor stay put (→ a blurry face lands
next to the wrong person). So we do it in stages:

1. **Anchors** — only high-quality faces (confident detection, big enough) define the
   identities. They are clustered with average-linkage cosine.
2. **Centroid merge** — identities whose centroids are very close are fused, healing a
   person who got split across a few fragments.
3. **Assignment with a margin** — every remaining (weak) face is attached to the nearest
   identity *only if* it is both close enough AND clearly closer than the runner-up.
   Otherwise it goes to the **unsorted** pile instead of contaminating a person.

Everything runs on the cached embeddings, so re-grouping with new settings is instant —
no re-detection.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .config import (
    CLUSTER_ASSIGN_DISTANCE,
    CLUSTER_DISTANCE,
    CLUSTER_MARGIN,
    CLUSTER_MERGE_DISTANCE,
    CLUSTER_MIN_DET,
    CLUSTER_MIN_FACE_PX,
)
from .store import ClusterRec, Job


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-8)


def _agglomerate(mat: np.ndarray, distance: float) -> np.ndarray:
    if len(mat) == 1:
        return np.array([0])
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance, metric="cosine", linkage="average"
    )
    return model.fit_predict(mat)


def recluster(
    job: Job,
    *,
    min_det_score: float = CLUSTER_MIN_DET,
    min_face_px: int = CLUSTER_MIN_FACE_PX,
    distance: float = CLUSTER_DISTANCE,
    merge_distance: float = CLUSTER_MERGE_DISTANCE,
    assign_distance: float = CLUSTER_ASSIGN_DISTANCE,
    margin: float = CLUSTER_MARGIN,
) -> None:
    """Rebuild job.clusters (and job.unsorted) from cached embeddings, in place."""
    job.params = {
        "min_det_score": min_det_score,
        "min_face_px": min_face_px,
        "distance": distance,
        "merge_distance": merge_distance,
        "assign_distance": assign_distance,
        "margin": margin,
    }

    faces = job.faces
    ids = np.array(sorted(faces))
    if len(ids) == 0:
        job.clusters, job.unsorted, job.num_clusters = {}, [], 0
        return

    E = job.embeddings[ids]  # row k ↔ face id ids[k]
    det = np.array([faces[i].det_score for i in ids])
    size = np.array(
        [min(faces[i].bbox[2] - faces[i].bbox[0], faces[i].bbox[3] - faces[i].bbox[1]) for i in ids]
    )

    anchor_mask = (det >= min_det_score) & (size >= min_face_px)
    if anchor_mask.sum() < 2:  # not enough good faces — treat everything as an anchor
        anchor_mask = np.ones(len(ids), dtype=bool)
    anchor_idx = np.where(anchor_mask)[0]

    # 1) cluster anchors
    labels = _agglomerate(E[anchor_idx], distance)
    groups: dict[int, list[int]] = {}
    for k, lab in zip(anchor_idx, labels):
        groups.setdefault(int(lab), []).append(int(k))
    centroids = _normalize(np.vstack([E[groups[l]].mean(0) for l in sorted(groups)]))

    # 2) merge near-duplicate identities (a person split into fragments)
    if merge_distance and len(groups) > 1:
        merge_labels = _agglomerate(centroids, merge_distance)
        merged: dict[int, list[int]] = {}
        for old_lab, new_lab in zip(sorted(groups), merge_labels):
            merged.setdefault(int(new_lab), []).extend(groups[old_lab])
        groups = {i: members for i, members in enumerate(merged.values())}
        centroids = _normalize(np.vstack([E[groups[l]].mean(0) for l in sorted(groups)]))

    # label→col map is identity now (groups keyed 0..K-1, centroids row j ↔ group j)
    assign: dict[int, int] = {k: lab for lab, members in groups.items() for k in members}

    # 3) assign weak faces with a margin, else unsorted
    unsorted_k: list[int] = []
    nonanchor = np.where(~anchor_mask)[0]
    if len(nonanchor):
        dist = 1.0 - (E[nonanchor] @ centroids.T)  # (M, K) cosine distances
        if dist.shape[1] >= 2:
            top2 = np.argsort(dist, axis=1)[:, :2]
            for row, k in enumerate(nonanchor):
                c0, c1 = top2[row]
                d1, d2 = dist[row, c0], dist[row, c1]
                if d1 <= assign_distance and (d2 - d1) >= margin:
                    assign[int(k)] = int(c0)
                else:
                    unsorted_k.append(int(k))
        else:  # only one identity — attach if close enough, no margin possible
            for row, k in enumerate(nonanchor):
                if dist[row, 0] <= assign_distance:
                    assign[int(k)] = 0
                else:
                    unsorted_k.append(int(k))

    # materialize clusters, numbered by photo count (most photographed person first)
    final: dict[int, list[int]] = {}
    for k, lab in assign.items():
        final.setdefault(lab, []).append(int(ids[k]))

    ordered = sorted(final.values(), key=lambda fs: len({faces[f].photo_id for f in fs}), reverse=True)
    job.clusters = {}
    for new_id, members in enumerate(ordered):
        members.sort(key=lambda f: faces[f].det_score, reverse=True)
        for f in members:
            faces[f].cluster_id = new_id
        photo_ids = sorted({faces[f].photo_id for f in members})
        job.clusters[new_id] = ClusterRec(
            id=new_id, face_ids=members, photo_ids=photo_ids, rep_face_id=members[0]
        )

    job.unsorted = [int(ids[k]) for k in unsorted_k]
    for f in job.unsorted:
        faces[f].cluster_id = -1
    job.num_clusters = len(job.clusters)
    apply_names(job)


def apply_names(job: Job) -> None:
    """Re-derive each cluster's name from job.names (face_id → name).

    Names are anchored to faces, not cluster ids, so they follow a person across a
    re-group. A cluster takes the name of its representative (or any named member)."""
    if not job.names:
        return
    for cluster in job.clusters.values():
        if cluster.rep_face_id in job.names:
            cluster.name = job.names[cluster.rep_face_id]
            continue
        for f in cluster.face_ids:
            if f in job.names:
                cluster.name = job.names[f]
                break


# Back-compat name used by the scan pipeline.
def cluster_job(job: Job) -> None:
    recluster(job)


def cluster_centroids(job: Job) -> dict[int, np.ndarray]:
    """Unit-norm centroid per cluster (row aligned to cluster id)."""
    cents = {}
    for cid, c in job.clusters.items():
        v = job.embeddings[c.face_ids].mean(0)
        cents[cid] = v / (np.linalg.norm(v) + 1e-8)
    return cents


def rank_clusters_by_selfie(job: Job, query: np.ndarray, top_faces: int = 3) -> list[dict]:
    """Score every cluster against a selfie embedding (mean of its top-k face sims)."""
    if job.embeddings is None or not job.clusters:
        return []
    q = query.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    results: list[dict] = []
    for cid, cluster in job.clusters.items():
        sims = job.embeddings[cluster.face_ids] @ q
        sims.sort()
        best = sims[-top_faces:] if sims.size > top_faces else sims
        results.append({"cluster_id": cid, "score": float(best.mean())})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
