// MANO forward pass in plain JS (zero betas), matching smplx.lbs: pose correctives + linear blend skinning.
// model: parsed mano_right.json; pose: 48 numbers (global orient + 15 joints, axis-angle, radians,
// WITHOUT pose_mean - it is added here like smplx with flat_hand_mean=False); transl: [x, y, z].
// Writes 778*3 vertex coordinates into out (Float32Array) and returns it.
function manoPrepare(model) {
  const nV = model.v_template.length / 3, nJ = model.parents.length;
  return Object.assign(model, {
    nV, nJ,
    _vt: Float64Array.from(model.v_template), _J: Float64Array.from(model.J),
    _pd: Float32Array.from(model.posedirs), _w: Float32Array.from(model.weights),
    _vp: new Float64Array(nV * 3), _R: new Float64Array(nJ * 9), _G: new Float64Array(nJ * 12), _A: new Float64Array(nJ * 12),
  });
}

function manoRodrigues(x, y, z, R, o) {
  const t = Math.sqrt(x * x + y * y + z * z + 1e-16), c = Math.cos(t), s = Math.sin(t), C = 1 - c;
  const a = x / t, b = y / t, d = z / t;
  R[o] = c + a * a * C;     R[o + 1] = a * b * C - d * s; R[o + 2] = a * d * C + b * s;
  R[o + 3] = b * a * C + d * s; R[o + 4] = c + b * b * C; R[o + 5] = b * d * C - a * s;
  R[o + 6] = d * a * C - b * s; R[o + 7] = d * b * C + a * s; R[o + 8] = c + d * d * C;
}

function manoForward(m, pose, transl, out) {
  const { nV, nJ, _vt: vt, _J: J, _pd: pd, _w: w, _vp: vp, _R: R, _G: G, _A: A, parents, pose_mean } = m;
  for (let j = 0; j < nJ; j++) {
    manoRodrigues(pose[j * 3] + pose_mean[j * 3], pose[j * 3 + 1] + pose_mean[j * 3 + 1], pose[j * 3 + 2] + pose_mean[j * 3 + 2], R, j * 9);
  }
  // pose correctives: feature = (R_j - I) for j >= 1, flattened row-major (135)
  vp.set(vt);
  const nF = (nJ - 1) * 9, n3 = nV * 3;
  for (let k = 0; k < nF; k++) {
    const j = 1 + Math.floor(k / 9), e = k % 9;
    const f = R[j * 9 + e] - (e === 0 || e === 4 || e === 8 ? 1 : 0);
    if (f === 0) continue;
    const row = k * n3;
    for (let i = 0; i < n3; i++) vp[i] += f * pd[row + i];
  }
  // kinematic chain: G_j = G_parent * [R_j | J_j - J_parent]
  for (let j = 0; j < nJ; j++) {
    const p = parents[j], r = j * 9, g = j * 12;
    const tx = J[j * 3] - (p < 0 ? 0 : J[p * 3]), ty = J[j * 3 + 1] - (p < 0 ? 0 : J[p * 3 + 1]), tz = J[j * 3 + 2] - (p < 0 ? 0 : J[p * 3 + 2]);
    if (p < 0) {
      for (let a = 0; a < 3; a++) { G[g + a * 4] = R[r + a * 3]; G[g + a * 4 + 1] = R[r + a * 3 + 1]; G[g + a * 4 + 2] = R[r + a * 3 + 2]; }
      G[g + 3] = tx; G[g + 7] = ty; G[g + 11] = tz;
    } else {
      const q = p * 12;
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) G[g + a * 4 + b] = G[q + a * 4] * R[r + b] + G[q + a * 4 + 1] * R[r + 3 + b] + G[q + a * 4 + 2] * R[r + 6 + b];
        G[g + a * 4 + 3] = G[q + a * 4] * tx + G[q + a * 4 + 1] * ty + G[q + a * 4 + 2] * tz + G[q + a * 4 + 3];
      }
    }
    // A_j = G_j with translation t - R J_j (removes the rest joint position)
    for (let a = 0; a < 3; a++) {
      A[g + a * 4] = G[g + a * 4]; A[g + a * 4 + 1] = G[g + a * 4 + 1]; A[g + a * 4 + 2] = G[g + a * 4 + 2];
      A[g + a * 4 + 3] = G[g + a * 4 + 3] - (G[g + a * 4] * J[j * 3] + G[g + a * 4 + 1] * J[j * 3 + 1] + G[g + a * 4 + 2] * J[j * 3 + 2]);
    }
  }
  for (let i = 0; i < nV; i++) {
    const x = vp[i * 3], y = vp[i * 3 + 1], z = vp[i * 3 + 2];
    let ox = 0, oy = 0, oz = 0;
    for (let j = 0; j < nJ; j++) {
      const wt = w[i * nJ + j];
      if (wt === 0) continue;
      const g = j * 12;
      ox += wt * (A[g] * x + A[g + 1] * y + A[g + 2] * z + A[g + 3]);
      oy += wt * (A[g + 4] * x + A[g + 5] * y + A[g + 6] * z + A[g + 7]);
      oz += wt * (A[g + 8] * x + A[g + 9] * y + A[g + 10] * z + A[g + 11]);
    }
    out[i * 3] = ox + transl[0]; out[i * 3 + 1] = oy + transl[1]; out[i * 3 + 2] = oz + transl[2];
  }
  return out;
}

// .npz reader (np.savez / np.savez_compressed): returns {name: {shape, data}}
async function readNpz(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf);
  let eocd = -1;
  for (let i = buf.byteLength - 22; i >= Math.max(0, buf.byteLength - 70000); i--) if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('not a .npz (zip) file');
  let n = dv.getUint16(eocd + 10, true), cd = dv.getUint32(eocd + 16, true);
  const out = {};
  for (let k = 0; k < n; k++) {
    const method = dv.getUint16(cd + 10, true), csize = dv.getUint32(cd + 20, true);
    const nameLen = dv.getUint16(cd + 28, true), extraLen = dv.getUint16(cd + 30, true), commentLen = dv.getUint16(cd + 32, true);
    const local = dv.getUint32(cd + 42, true);
    const name = new TextDecoder().decode(u8.subarray(cd + 46, cd + 46 + nameLen));
    cd += 46 + nameLen + extraLen + commentLen;
    const lNameLen = dv.getUint16(local + 26, true), lExtra = dv.getUint16(local + 28, true);
    let data = u8.slice(local + 30 + lNameLen + lExtra, local + 30 + lNameLen + lExtra + csize);
    if (method === 8) data = new Uint8Array(await new Response(new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer());
    else if (method !== 0) throw new Error(`${name}: unsupported zip compression ${method}`);
    out[name.replace(/\.npy$/, '')] = parseNpy(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
  }
  return out;
}

function parseNpy(buf) {
  const u8 = new Uint8Array(buf), dv = new DataView(buf);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY') throw new Error('not a .npy array');
  const major = u8[6], hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true), hoff = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.subarray(hoff, hoff + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1], fortran = /'fortran_order':\s*True/.test(header);
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)[1].match(/\d+/g) || []).map(Number);
  const body = buf.slice(hoff + hlen), size = shape.reduce((a, b) => a * b, 1);
  const types = { '<f4': Float32Array, '<f8': Float64Array, '<i4': Int32Array, '<i8': BigInt64Array, '|u1': Uint8Array, '|b1': Uint8Array, '<f2': null };
  const T = types[descr];
  if (!T) throw new Error(`unsupported dtype ${descr}`);
  let data = Array.from(new T(body, 0, size), Number);
  if (fortran && shape.length === 2) { const [r, c] = shape, t = new Array(size); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) t[i * c + j] = data[j * r + i]; data = t; }
  return { shape, data };
}

// Find right/left pose (T,48) and trans (T,3) in whatever layout the .npz uses.
function pickHands(arrs) {
  const rows = a => { let s = a.shape.slice(); while (s.length > 2 && s[0] === 1) s.shift(); return s.length === 2 ? { T: s[0], D: s[1], d: a.data } : null; };
  const get = (...names) => { for (const n of names) if (arrs[n]) { const r = rows(arrs[n]); if (r) return r; } return null; };
  const cols = (r, a, b) => { const o = new Float32Array(r.T * (b - a)); for (let t = 0; t < r.T; t++) for (let k = a; k < b; k++) o[t * (b - a) + k - a] = r.d[t * r.D + k]; return o; };
  const res = {};
  for (const h of ['right', 'left']) {
    const p = get(`${h}_pose`, `${h}_pose_pred`, `pose_${h}`, `${h}_poses`), t = get(`${h}_trans`, `${h}_trans_pred`, `trans_${h}`);
    if (p && p.D === 48 && t && t.D === 3) { res[h] = { T: p.T, pose: cols(p, 0, 48), trans: cols(t, 0, 3) }; continue; }
    const c = get(h, `${h}_hand`, `pose_${h}`);           // combined rows
    if (c && c.D === 51) { res[h] = { T: c.T, trans: cols(c, 0, 3), pose: cols(c, 3, 51) }; continue; }
    if (c && c.D >= 52) { res[h] = { T: c.T, trans: cols(c, 1, 4), pose: cols(c, 4, 52) }; continue; }   // PianoMotion10M rows
  }
  if (!res.right || !res.left) {
    const pr = get('prediction', 'pose_hat', 'pred'), g = get('guide', 'trans', 'translation');
    if (pr && pr.D === 96 && g && g.D === 6) {
      res.right = { T: pr.T, pose: cols(pr, 0, 48), trans: cols(g, 0, 3) };
      res.left = { T: pr.T, pose: cols(pr, 48, 96), trans: cols(g, 3, 6) };
    }
  }
  if (!res.right || !res.left) {
    const list = Object.entries(arrs).map(([k, a]) => `${k} ${JSON.stringify(a.shape)}`).join(', ');
    throw new Error(`could not find right/left pose (T,48) + trans (T,3). Arrays in file: ${list}`);
  }
  const fps = arrs.fps ? Number(arrs.fps.data[0]) : 30;
  return { right: res.right, left: res.left, T: Math.min(res.right.T, res.left.T), fps };
}

if (typeof module !== 'undefined') module.exports = { manoPrepare, manoForward, readNpz, parseNpy, pickHands };
