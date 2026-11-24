"""Build the outfit maker's dress form as ONE seamless surface.

The body (head, neck, torso, shoulders, arms, hands, legs, feet) is described as a signed distance
field: each part is a shape, and the parts are joined with a smooth union so they flow into each
other with no seams. The field is sampled on a fine grid, turned into a mesh (marching cubes),
simplified, and given normals taken straight from the field, so shading is smooth everywhere.

Shape numbers (profiles, bust, legs, arms) are read from outfit.html, so the clothes, which are
built in the page from the same numbers, keep fitting. Rerun this after changing them:

    python3 tools/build_mannequin.py           # writes binder/mannequin.bin

Needs numpy<2, scikit-image<0.23 and fast-simplification<0.1.8 (Python 3.9+).
"""
import ast
import os
import re
import struct
import sys
import time

import numpy as np
from skimage.measure import marching_cubes
import fast_simplification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'outfit.html')
OUT = os.path.join(ROOT, 'binder', 'mannequin.bin')

html = open(SRC, encoding='utf-8').read()


def js_value(name):
    """Read `const NAME = <literal>` (number, array or {key: value} object) from outfit.html."""
    m = re.search(r'const ' + name + r'\s*=\s*', html)
    if not m:
        sys.exit('not found in outfit.html: ' + name)
    i = m.end()
    if html.startswith('smooth(', i):
        i += len('smooth(')
    depth, j = 0, i
    while True:
        c = html[j]
        if c in '[{(':
            depth += 1
        elif c in ']})':
            depth -= 1
            if depth == 0:
                j += 1
                break
        elif depth == 0 and c in ',;':
            break
        j += 1
    text = html[i:j]
    text = re.sub(r'//[^\n]*', '', text)
    text = re.sub(r'([{,]\s*)([A-Za-z_]\w*)\s*:', r'\1"\2":', text)   # quote object keys
    return ast.literal_eval(text.strip())


def js_number(name):
    m = re.search(r'\b' + name + r'\s*=\s*([-\d.]+)', html)
    if not m:
        sys.exit('not found in outfit.html: ' + name)
    return float(m.group(1))


# ---------- shape numbers from the page ----------
ZS = js_number('ZS')
CROTCH = js_number('CROTCH')
LEG_X, LEG_TOP = js_number('LEG_X'), js_number('LEG_TOP')
ARM_ANGLE = js_number('ARM_ANGLE')
m = re.search(r'SHOULDER = new THREE\.Vector3\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', html)
SHOULDER = tuple(float(v) for v in m.groups())
BUST, CHEST, LUMBAR, SEAT, BELLY = (js_value(n) for n in ('BUST', 'CHEST', 'LUMBAR', 'SEAT', 'BELLY'))
LEG_ROT = js_value('LEG_ROT')   # {right: [rx, rz], left: [rx, rz]}


def catmull(t, p0, p1, p2, p3):
    v0, v1 = (p2 - p0) * 0.5, (p3 - p1) * 0.5
    t2 = t * t
    return (2 * p1 - 2 * p2 + v0 + v1) * t2 * t + (-3 * p1 + 3 * p2 - 2 * v0 - v1) * t2 + v0 * t + p1


def smooth(pts, n):
    """Same as THREE.SplineCurve(points).getPoints(n)."""
    P = np.array(pts, dtype=float)
    out = []
    L = len(P)
    for d in range(n + 1):
        p = (L - 1) * d / n
        i = int(np.floor(p))
        w = p - i
        p0 = P[i if i == 0 else i - 1]
        p1 = P[i]
        p2 = P[L - 1 if i > L - 2 else i + 1]
        p3 = P[L - 1 if i > L - 3 else i + 2]
        out.append((catmull(w, p0[0], p1[0], p2[0], p3[0]), catmull(w, p0[1], p1[1], p2[1], p3[1])))
    return np.array(out)


def lookup(pts):
    ys, rs = pts[:, 1], pts[:, 0]
    order = np.argsort(ys, kind='stable')
    ys, rs = ys[order], rs[order]
    return lambda y: np.interp(y, ys, rs)


def profile(name, n):
    m = re.search(r'const ' + name + r' = smooth\((\[.*?\]\]), (\d+)\)', html, re.S)
    if not m:
        sys.exit('profile not found: ' + name)
    text = re.sub(r'//[^\n]*', '', m.group(1))
    return smooth(ast.literal_eval(text), int(m.group(2)))


torsoR = lookup(profile('torsoPts', 260))
legR = lookup(profile('legPts', 140))
armR = lookup(profile('armPts', 120))


# ---------- helpers (match the page) ----------
def ease(t):
    return t * t * (3 - 2 * t)


def ramp(y, a, b):
    return ease(np.clip((y - a) / (b - a), 0, 1))


def SH(y):
    return 0.05 * np.exp(-(((y - 1.425) / 0.045) ** 2))


def depth_at(y):
    return ZS * (1 - 0.28 * (1 - ramp(y, 0.84, 1.0))) + (1 - ZS) * ramp(y, 1.46, 1.53) + 0.12 * ramp(y, 1.56, 1.64)


def head_fwd(y):
    return 0.012 * ramp(y, 1.57, 1.66)


def ray_out(sx, cz, cx, cz0, rx, rz):
    """Distance from the axis to where a ray (sx, cz) leaves an oval centred at (cx, cz0)."""
    ok = (rx > 0) & (rz > 0)
    rx = np.where(ok, rx, 1)
    rz = np.where(ok, rz, 1)
    a = np.maximum((sx / rx) ** 2 + (cz / rz) ** 2, 1e-12)   # on the axis itself the direction is undefined
    b = -2 * (sx * cx / (rx * rx) + cz * cz0 / (rz * rz))
    c = (cx / rx) ** 2 + (cz0 / rz) ** 2 - 1
    disc = b * b - 4 * a * c
    far = (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a)
    return np.where(ok & (disc >= 0), np.maximum(far, 0), 0)


def smax(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.maximum(a, b) + h * h * k / 4


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0) / k
    return np.minimum(a, b) - h * h * k / 4


def bump(bp, x, y):
    # same as the page: two soft mounds at +-x added together, so no crease on the centre line
    sy = np.where((y > bp['y']) & ('syUp' in bp), bp.get('syUp', bp['sy']), bp['sy'])
    vy = np.exp(-(((y - bp['y']) / sy) ** 2))
    if not bp['x']:
        return bp['a'] * vy * np.exp(-((x / bp['sx']) ** 2))
    g = lambda c: np.exp(-(((x - c) / bp['sx']) ** 2))
    return bp['a'] * vy * (g(bp['x']) + g(-bp['x'])) / (1 + np.exp(-((2 * bp['x'] / bp['sx']) ** 2)))


def curve_out(x, y, z):
    r = np.hypot(x, z) + 1e-9
    nz = z / r
    front = (bump(BUST, x, y) + bump(CHEST, x, y) + bump(BELLY, x, y)) * ease(np.clip(nz, 0, 1))
    back = (bump(SEAT, x, y) + bump(LUMBAR, x, y)) * ease(np.clip(-nz, 0, 1))
    return np.where(nz > 0, front, back)


def rot_z(x, y, a):
    c, s = np.cos(a), np.sin(a)
    return x * c - y * s, x * s + y * c


def capsule(px, py, pz, ya, yb, r):
    """Distance to a vertical capsule whose core runs from y=yb (bottom) to y=ya (top) on the y axis."""
    q = np.clip(py, yb, ya)
    return np.sqrt(px * px + (py - q) ** 2 + pz * pz) - r


# ---------- the body as a distance field ----------
def torso(x, y, z):
    rho = np.hypot(x, z) + 1e-9
    sx, cz = x / rho, z / rho
    t = torsoR(y)
    R = np.where(y > CROTCH, ray_out(sx, cz, 0.0, head_fwd(y), t + SH(y), t * depth_at(y)), 0)
    d = rho - R - curve_out(x, y, z)
    # navel: a small soft dimple on the front, a little below the waist
    front = ease(np.clip(z / rho, 0, 1)) ** 4
    d = d + 0.0045 * np.exp(-((x / 0.0055) ** 2 + ((y - NAVEL_Y) / 0.0075) ** 2)) * front
    return np.maximum(smax(d, CROTCH - y, 0.045), y - 1.813)   # the crotch rounds into an arch, not a flat edge


def ellipsoid(px, py, pz, rx, ry, rz):
    k0 = np.sqrt((px / rx) ** 2 + (py / ry) ** 2 + (pz / rz) ** 2)
    k1 = np.sqrt((px / rx ** 2) ** 2 + (py / ry ** 2) ** 2 + (pz / rz ** 2) ** 2) + 1e-12
    return k0 * (k0 - 1) / k1


def leg(x, y, z, side):
    # into the leg's own frame: the leg group is rotated X then Z (Euler XYZ) about the hip,
    # so undo the X rotation first, then the Z rotation
    rxa, rza = LEG_ROT['right'] if side > 0 else LEG_ROT['left']
    px, py, pz = x - side * LEG_X, y - LEG_TOP, z
    c, s = np.cos(-rxa), np.sin(-rxa)
    py, pz = py * c - pz * s, py * s + pz * c
    lx, ly = rot_z(px, py, -rza)
    ly = ly + LEG_TOP
    lz = pz
    rho = np.hypot(lx, lz) + 1e-9
    sx, cz = lx / rho, lz / rho
    R = np.where(ly >= 0.05, legR(ly), 0)
    # above the hip joint the leg fades away smoothly (its surface sinks inward), so it never ends with an edge
    d = rho - ray_out(sx, cz, 0.0, 0.0, R, R * 0.92) + 0.9 * np.maximum(ly - 0.94, 0) ** 1.6
    d = np.maximum(d, 0.04 - ly)
    # arched shop-mannequin foot: heel lifted, toes pointing down to the base
    a = 0.38
    fy, fz = ly - 0.07, lz - 0.038
    qy, qz = fy * np.cos(a) - fz * np.sin(a), fy * np.sin(a) + fz * np.cos(a)
    foot = ellipsoid(lx, qy, qz, 0.027, 0.021, 0.06)
    return smin(d, foot, 0.018)


# Thickness (front-to-back of the hand) and width of the forearm-wrist-palm, as smooth curves down the arm.
_tt = [0.40, 0.46, 0.52, 0.60, 0.625, 0.66, 0.684, 0.70]
_r40, _r46 = float(armR(-0.40)), float(armR(-0.46))
HAND_RX = lookup(smooth([[v, t] for v, t in zip([_r40, _r46, 0.0245, 0.0165, 0.0115, 0.0105, 0.009, 0.004], _tt)], 200))
HAND_RZ = lookup(smooth([[v, t] for v, t in zip([_r40, _r46, 0.026, 0.0205, 0.0215, 0.0225, 0.0205, 0.008], _tt)], 200))

NAVEL_Y = 1.122

FINGERS = [(0.0145, 0.03), (0.005, 0.037), (-0.0045, 0.034), (-0.0135, 0.027)]


def arm(x, y, z, side):
    ax, ay = rot_z(x - side * SHOULDER[0], y - SHOULDER[1], -side * ARM_ANGLE)
    az = z
    # arm, wrist and palm as one shape: an oval cross-section that is round down the arm, then narrows at the
    # wrist and flattens into the palm (thin front-to-back of the hand, wide across it)
    t = -ay
    round_r = armR(ay)
    rx = np.where(t < 0.40, round_r, HAND_RX(t))
    rz = np.where(t < 0.40, round_r, HAND_RZ(t))
    rmin = np.maximum(np.minimum(rx, rz), 1e-4)
    q = np.sqrt((ax / np.maximum(rx, 1e-4)) ** 2 + (az / np.maximum(rz, 1e-4)) ** 2)
    d = np.maximum((q - 1) * rmin, np.maximum(ay, t - 0.70))
    d = smin(d, np.sqrt(ax * ax + (ay + 0.004) ** 2 + az * az) - 0.038, 0.012)   # round top of the arm (deltoid), blended in
    palm = d
    hand = palm
    fr = 0.0068
    for fz, ln in FINGERS:
        fx, fy = rot_z(ax, ay + 0.662, side * 0.16)                         # undo the finger's curl
        f = capsule(fx / 0.9, fy, az - fz, -fr, -(ln + fr), fr) * 0.9
        hand = smin(hand, f, 0.005)
    # thumb: Euler XYZ (-0.35, 0, -side*0.18) about its base
    # undo it in reverse order: X first, then Z
    tx, ty, tz = ax + side * 0.004, ay + 0.62, az - 0.014
    c, s = np.cos(0.35), np.sin(0.35)
    ty, tz = ty * c - tz * s, ty * s + tz * c
    tx, ty = rot_z(tx, ty, side * 0.18)
    thumb = capsule(tx, ty, tz, -0.009, -0.033, 0.0075)
    hand = smin(hand, thumb, 0.006)
    return hand


def body(x, y, z):
    t = torso(x, y, z)
    # thighs round into each other only right at the crotch, so there's no sharp notch but still a gap below
    legs = smin(leg(x, y, z, 1), leg(x, y, z, -1), 0.001 + 0.014 * ramp(y, 0.8, 0.87))   # just a soft crotch, the thighs stay apart
    d = smin(t, legs, 0.03 + 0.05 * ramp(y, 0.86, 0.97))   # wide, soft blend at the hips; tight at the crotch so it stays an arch
    arms = np.minimum(arm(x, y, z, 1), arm(x, y, z, -1))
    # a wide, soft join at the shoulder, narrowing below the armpit so the arm doesn't web to the side
    k = 0.012 + 0.038 * ramp(y, 1.3, 1.41)
    return smin(d, arms, k)


# ---------- sample, mesh, simplify ----------
def main():
    h = float(os.environ.get('MANNEQUIN_STEP', '0.0025'))
    lo = np.array([-0.31, 0.025, -0.16])
    hi = np.array([0.31, 1.825, 0.19])
    n = np.ceil((hi - lo) / h).astype(int) + 1
    xs = lo[0] + np.arange(n[0]) * h
    ys = lo[1] + np.arange(n[1]) * h
    zs = lo[2] + np.arange(n[2]) * h
    print('grid', n, '=', int(np.prod(n)), 'samples')
    t0 = time.time()
    vol = np.empty((n[0], n[1], n[2]), dtype=np.float32)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    for j in range(0, n[1], 16):
        jj = np.arange(j, min(j + 16, n[1]))
        Y = ys[jj][None, :, None]
        vol[:, jj, :] = body(X[:, None, :], np.broadcast_to(Y, (n[0], len(jj), n[2])), Z[:, None, :])
    print('field %.1fs' % (time.time() - t0))
    verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(h, h, h))
    verts += lo
    print('marching cubes:', len(verts), 'verts', len(faces), 'faces')
    target = int(os.environ.get('MANNEQUIN_FACES', '110000'))
    reduction = max(0.0, 1 - target / len(faces))
    verts, faces = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int64), target_reduction=reduction)
    print('simplified:', len(verts), 'verts', len(faces), 'faces')
    # normals straight from the field: smooth, with no creases where parts meet
    e = 0.0007
    vx, vy, vz = verts[:, 0].astype(float), verts[:, 1].astype(float), verts[:, 2].astype(float)
    g = np.stack([
        body(vx + e, vy, vz) - body(vx - e, vy, vz),
        body(vx, vy + e, vz) - body(vx, vy - e, vz),
        body(vx, vy, vz + e) - body(vx, vy, vz - e),
    ], axis=1)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    # make every face wind outward (agreeing with the field's gradient)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    if np.mean(np.sum(fn * (g[faces[:, 0]] + g[faces[:, 1]] + g[faces[:, 2]]), axis=1) < 0) > 0.5:
        faces = faces[:, ::-1]
    # pack: header, int16 positions (quantised to the bounds), int8 normals, uint16/32 indices
    vmin, vmax = verts.min(0), verts.max(0)
    q = np.round((verts - vmin) / (vmax - vmin) * 65535 - 32768).astype(np.int16)
    nq = np.round(g * 127).astype(np.int8)
    idx_type = np.uint16 if len(verts) < 65536 else np.uint32
    with open(OUT, 'wb') as f:
        f.write(b'MANQ')
        f.write(struct.pack('<III', len(verts), faces.size, 2 if idx_type == np.uint16 else 4))
        f.write(struct.pack('<6f', *vmin, *vmax))
        f.write(q.tobytes())
        f.write(nq.tobytes())
        f.write(b'\0' * ((4 - f.tell() % 4) % 4))   # indices start on a 4-byte boundary
        f.write(faces.astype(idx_type).tobytes())
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
