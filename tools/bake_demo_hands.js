// Bake the browser demo (demo/piano-hands.html) into files for the Unity player.
//
//   node tools/bake_demo_hands.js OUT_DIR
//
// Needs Playwright with Chromium (npm i playwright). Writes:
//   demo_hands.hmotion  skinned hand meshes per frame, Unity coordinates (HandMotionPlayer.cs)
//   demo_notes.json     notes + keyboard layout in Unity metres (PianoKeyboard.cs)
//   demo_audio.wav      synthesised piano audio, same timing (AudioSource drives the clock)
//
// Set THREE_DIR=<path to node_modules/three> to serve three.js locally when the CDN is blocked.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const S = 0.0235;   // one white key = 2.35 cm; demo units -> metres
const FPS = 30;

(async () => {
  const outDir = process.argv[2] || 'demo_out';
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const T = process.env.THREE_DIR;
  if (T) {
    await page.route(/three(\.min)?\.js$/, r => r.fulfill({ path: path.join(T, 'build/three.min.js'), contentType: 'text/javascript' }));
    await page.route(/OrbitControls\.js$/, r => r.fulfill({ path: path.join(T, 'examples/js/controls/OrbitControls.js'), contentType: 'text/javascript' }));
    await page.route(/GLTFLoader\.js$/, r => r.fulfill({ path: path.join(T, 'examples/js/loaders/GLTFLoader.js'), contentType: 'text/javascript' }));
  }
  await page.route(/fonts\.(googleapis|gstatic)/, r => r.abort());
  await page.goto('file://' + path.resolve(__dirname, '../demo/piano-hands.html'));
  const d = await page.evaluate(fps => window.__bakeHands(fps), FPS);
  await browser.close();

  // demo units (y up, player at +z) -> Unity metres (y up, left-handed, camera looks +z): flip z.
  const [cx, cy, cz] = d.keyboard.center;
  const conv = arr => {
    const o = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i += 3) {
      o[i] = (arr[i] - cx) * S; o[i + 1] = (arr[i + 1] - cy) * S; o[i + 2] = -(arr[i + 2] - cz) * S;
    }
    return o;
  };
  const flipWinding = f => { const o = Int32Array.from(f); for (let i = 0; i < o.length; i += 3) [o[i + 1], o[i + 2]] = [o[i + 2], o[i + 1]]; return o; };
  const fr = flipWinding(d.faces.R), fl = flipWinding(d.faces.L);
  const nvR = d.verts.R[0].length / 3, nvL = d.verts.L[0].length / 3;

  const head = Buffer.alloc(32);
  head.write('HMOT', 0, 'ascii');
  head.writeInt32LE(1, 4); head.writeFloatLE(FPS, 8); head.writeInt32LE(d.frames, 12);
  head.writeInt32LE(nvR, 16); head.writeInt32LE(fr.length / 3, 20);
  head.writeInt32LE(nvL, 24); head.writeInt32LE(fl.length / 3, 28);
  const parts = [head, Buffer.from(fr.buffer), Buffer.from(fl.buffer)];
  for (let k = 0; k < d.frames; k++) {
    parts.push(Buffer.from(conv(d.verts.R[k]).buffer), Buffer.from(conv(d.verts.L[k]).buffer));
  }
  fs.writeFileSync(path.join(outDir, 'demo_hands.hmotion'), Buffer.concat(parts));

  fs.writeFileSync(path.join(outDir, 'demo_notes.json'), JSON.stringify({
    scale: S, x0: d.keyboard.x0, low: d.keyboard.low, high: d.keyboard.high, keyTravel: d.keyboard.keyTravel,
    center: [cx, cy, cz], notes: d.notes,
  }));

  // simple additive piano-ish synth, mono 16-bit 44.1 kHz
  const sr = 44100, dur = d.frames / FPS + 1, pcm = new Float32Array(Math.ceil(dur * sr));
  for (const n of d.notes) {
    const f = 440 * Math.pow(2, (n.pitch - 69) / 12), L = Math.max(0.25, n.off - n.on) + 0.3;
    const vol = n.pitch < 60 ? 0.12 : 0.16, start = Math.floor(n.on * sr), decay = n.pitch < 60 ? 1.4 : 0.9;
    for (let i = 0; i < L * sr && start + i < pcm.length; i++) {
      const t = i / sr;
      const env = Math.min(t / 0.008, 1) * (0.35 + 0.65 * Math.exp(-t / 0.12)) * Math.exp(-t / decay) * Math.min(1, (L - t) / 0.15);
      const tri = 2 / Math.PI * Math.asin(Math.sin(2 * Math.PI * f * t));
      pcm[start + i] += vol * env * (tri + 0.35 * Math.sin(4 * Math.PI * f * t) + 0.12 * Math.sin(6 * Math.PI * f * t));
    }
  }
  let peak = 1e-6; for (const x of pcm) peak = Math.max(peak, Math.abs(x));
  const wav = Buffer.alloc(44 + pcm.length * 2);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + pcm.length * 2, 4); wav.write('WAVE', 8);
  wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(sr, 24); wav.writeUInt32LE(sr * 2, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(pcm.length * 2, 40);
  for (let i = 0; i < pcm.length; i++) wav.writeInt16LE(Math.round(pcm[i] / peak * 0.8 * 32767), 44 + i * 2);
  fs.writeFileSync(path.join(outDir, 'demo_audio.wav'), wav);

  console.log(`${d.frames} frames @ ${FPS} fps, ${nvR}+${nvL} verts -> ${outDir}`);
})();
