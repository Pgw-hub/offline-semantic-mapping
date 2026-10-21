"""복셀 → OctoMap/RViz 스타일 정육면체 렌더 (rerun 0.16).

Boxes3D 는 와이어프레임만 그리므로, 속이 찬 큐브는 Mesh3D 로, 검은 모서리는 그 위에 얹는 Boxes3D 와이어로 만든다.
- cubes_mesh : 큐브당 면별 24정점(면마다 밝기 다름 → 입체감) · 12삼각형
- cubes_wire : 같은 위치의 검은 와이어프레임 (별도 자식 엔티티로 로깅해 토글 가능)
"""
import numpy as np
import rerun as rr

_CORNERS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32)
# (면의 정점 4개, 밝기) — 위 1.0 · 아래 0.5 · 옆면 두 방향 0.72/0.86 (RViz 큐브 리스트 느낌)
_FACES = [([4, 5, 6, 7], 1.00), ([0, 3, 2, 1], 0.50),
          ([0, 1, 5, 4], 0.72), ([2, 3, 7, 6], 0.72),
          ([1, 2, 6, 5], 0.86), ([3, 0, 4, 7], 0.86)]
_FACE_OFF = np.stack([_CORNERS[f] for f, _ in _FACES])                 # (6,4,3)
_SHADE = np.array([[s] * 4 for _, s in _FACES], np.float32)[..., None]  # (6,4,1)
_FACE_TRI = np.array([[0, 1, 2], [0, 2, 3]], np.uint32)


def cubes_mesh(centers, size, color, shade=True, alpha=1.0):
    """alpha<1 이면 RGBA 정점색. (rerun 0.16 뷰어의 메시 알파 지원은 제한적 — 점/선 알파는 동작)"""
    c = np.asarray(centers, np.float32).reshape(-1, 3); n = len(c); h = size / 2
    verts = (c[:, None, None, :] + _FACE_OFF[None] * h).reshape(-1, 3)                 # (n*24,3)
    face_base = (np.arange(6, dtype=np.uint32) * 4)[:, None, None]                     # (6,1,1)
    tris_cube = (_FACE_TRI[None] + face_base).reshape(-1, 3)                           # (12,3)
    tris = (tris_cube[None] + (np.arange(n, dtype=np.uint32) * 24)[:, None, None]).reshape(-1, 3)
    col = np.asarray(color, np.float32)[:3]
    fc = (col[None, None] * (_SHADE if shade else 1.0)).clip(0, 255).astype(np.uint8)  # (6,4,3)
    if alpha < 1.0:
        a = np.full((6, 4, 1), int(round(alpha * 255)), np.uint8); fc = np.concatenate([fc, a], -1)
    cols = np.broadcast_to(fc[None], (n,) + fc.shape).reshape(-1, fc.shape[-1])
    return rr.Mesh3D(vertex_positions=verts, triangle_indices=tris, vertex_colors=cols)


def cubes_wire(centers, size, color=(0, 0, 0), radius_frac=0.015, alpha=1.0):
    c = np.asarray(centers, np.float32).reshape(-1, 3)
    col = list(color)[:3] + ([int(round(alpha * 255))] if alpha < 1.0 else [])
    return rr.Boxes3D(centers=c, half_sizes=np.full((len(c), 3), size / 2, np.float32),
                      colors=[col], radii=size * radius_frac)


def log_cubes(entity, centers, size, color, edges=True, shade=True, alpha=1.0):
    """entity 에 큐브 메시, entity/wire 에 검은 모서리. alpha 는 둘 다에 적용."""
    rr.log(entity, cubes_mesh(centers, size, color, shade, alpha))
    if edges:
        rr.log(entity + "/wire", cubes_wire(centers, size, alpha=alpha))
