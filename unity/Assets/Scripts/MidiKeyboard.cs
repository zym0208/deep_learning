// Piano keyboard driven by the clip's MIDI: keys go down and light up while their note
// sounds. The colour says which hand is on the key at that moment (orange = right,
// blue = left); red means no fingertip is within `tolerance` keys of it, i.e. the motion
// misses the key. The top-left label shows, for every version (raw / refined / ...), the
// share of note onsets where a fingertip is on the right key.
//
// Where the keyboard goes: PianoMotion10M has no keyboard position, so with MANO hands
// (778 vertices) the keyboard is fitted to the fingertips at the note onsets, the same
// method as refiner/keypress.py: key k sits at o + u(k)*v (u = position in white keys),
// fingertips are assigned to the nearest key, o and v are refitted, repeat. Fit on the
// ground truth if you have it (put gt.hmotion first in Variant Files, fitVariant = 0).
// Without MANO hands (dummy.hmotion) set autoFit off and place it with the manual fields.
//
// Setup: notes file from
//   python tools/keypress_eval.py notes <data_root>/midi/<up>/<video>/<seq>.mid --duration 8 -o Assets/StreamingAssets/<seq>_notes.json
// then add this component to any GameObject, drag the HandMotionPlayer into Player and set
// Notes File. The keyboard is built as a child of the player, in the hands' space.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

public class MidiKeyboard : MonoBehaviour
{
    public HandMotionPlayer player;
    [Tooltip("File inside Assets/StreamingAssets, from tools/keypress_eval.py notes")]
    public string notesFile = "notes.json";
    [Tooltip("Seconds added to the clip time before looking up notes; positive = keys earlier.")]
    public float timeOffset = 0f;

    [Header("Placement")]
    [Tooltip("Fit the keyboard to the fingertips at note onsets (needs MANO hands).")]
    public bool autoFit = true;
    [Tooltip("Version used for the fit; the ground truth if you have it.")]
    public int fitVariant = 0;
    [Tooltip("Manual placement (player space): centre of middle C, and one white-key step.")]
    public Vector3 manualMiddleC = new Vector3(0, -0.05f, 0);
    public Vector3 manualStep = new Vector3(0.0235f, 0, 0);
    public int lowNote = 21, highNote = 108;

    [Header("Look")]
    [Tooltip("A fingertip within this many key widths (along the keyboard) counts as on the key.")]
    public float tolerance = 0.5f;
    public Color whiteColor = new Color(0.97f, 0.96f, 0.93f);
    public Color blackColor = new Color(0.09f, 0.09f, 0.11f);
    public Color rightColor = new Color(0.95f, 0.55f, 0.15f);
    public Color leftColor = new Color(0.25f, 0.55f, 0.95f);
    public Color missColor = new Color(0.9f, 0.15f, 0.15f);

    static readonly int[] TipIds = { 744, 320, 443, 554, 671 };   // MANO thumb..pinky
    const int ManoVerts = 778;

    [Serializable] class Note { public int pitch; public float on; public float off; public int velocity; }
    [Serializable] class NoteFile { public Note[] notes; }
    class Key { public Transform tr; public Vector3 restPos; public Renderer rend; public bool black; }

    Note[] _notes;
    readonly Dictionary<int, Key> _keys = new Dictionary<int, Key>();
    Material _white, _black, _right, _left, _miss;
    bool _built;
    Vector2 _o, _v;                     // keyboard line in player-space x/y
    float _depth;                       // key top surface (player-space z)
    string _status = "";
    readonly List<string> _hitText = new List<string>();

    static bool IsBlack(int m) { int k = ((m % 12) + 12) % 12; return k == 1 || k == 3 || k == 6 || k == 8 || k == 10; }

    public static float KeyCoord(int pitch)
    {
        int d = pitch - 60, oct = Mathf.FloorToInt(d / 12f), pc = d - oct * 12;
        float[] w = { 0, 0.5f, 1, 1.5f, 2, 3, 3.5f, 4, 4.5f, 5, 5.5f, 6 };
        return oct * 7 + w[pc];
    }

    // If the scene has no MidiKeyboard but StreamingAssets has <clip>_notes.json for the clip being
    // played (e.g. BV1YL4y1F77j_seq_0001_notes.json next to BV1YL4y1F77j_seq_0001_gt.hmotion),
    // add one automatically, so nothing has to be set up by hand.
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void AutoSetup()
    {
        var p = FindObjectOfType<HandMotionPlayer>();
        if (p == null || FindObjectOfType<MidiKeyboard>() != null) return;
        var first = p.variantFiles != null && p.variantFiles.Length > 0 ? p.variantFiles[0] : p.fileName;
        var stem = Path.GetFileNameWithoutExtension(first);
        int cut = stem.LastIndexOf('_');
        foreach (var cand in new[] { stem + "_notes.json", cut > 0 ? stem.Substring(0, cut) + "_notes.json" : null })
        {
            if (cand == null || !File.Exists(Path.Combine(Application.streamingAssetsPath, cand))) continue;
            var k = new GameObject("MidiKeyboard").AddComponent<MidiKeyboard>();
            k.player = p; k.notesFile = cand;
            return;
        }
    }

    void Start()
    {
        var path = Path.Combine(Application.streamingAssetsPath, notesFile);
        _notes = JsonUtility.FromJson<NoteFile>(File.ReadAllText(path)).notes ?? new Note[0];
        var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
        _white = new Material(shader) { color = whiteColor };
        _black = new Material(shader) { color = blackColor };
        _right = new Material(shader) { color = rightColor };
        _left = new Material(shader) { color = leftColor };
        _miss = new Material(shader) { color = missColor };
    }

    // ------------------------------------------------------------ fingertips

    bool HasMano => player.GetVertices(true, 0) != null && player.GetVertices(true, 0).Length == ManoVerts
                    && player.GetVertices(false, 0).Length == ManoVerts;

    // 10 fingertips (right thumb..pinky, then left) at a frame, player space.
    Vector3[] Tips(int frame, int variant)
    {
        var r = player.GetVertices(true, frame, variant);
        var l = player.GetVertices(false, frame, variant);
        var t = new Vector3[10];
        for (int i = 0; i < 5; i++) { t[i] = r[TipIds[i]]; t[5 + i] = l[TipIds[i]]; }
        return t;
    }

    struct Onset { public int pitch, frame; }

    List<Onset> Onsets()
    {
        var list = new List<Onset>();
        foreach (var n in _notes)
        {
            int f = Mathf.RoundToInt(n.on * player.Fps);
            if (n.on >= 0 && f < player.FrameCount) list.Add(new Onset { pitch = n.pitch, frame = f });
        }
        return list;
    }

    // nearest fingertip along the keyboard; returns its index, distance in key widths
    static int Nearest(Vector2[] c, Vector2 centre, Vector2 v, out float dist)
    {
        float step = v.magnitude; Vector2 dir = v / step;
        int best = 0; dist = float.MaxValue;
        for (int i = 0; i < c.Length; i++)
        {
            float d = Mathf.Abs(Vector2.Dot(c[i] - centre, dir)) / step;
            if (d < dist) { dist = d; best = i; }
        }
        return best;
    }

    // ------------------------------------------------------------ keyboard fit (see refiner/keypress.py)

    static void FitLine(List<float> u, List<Vector2> p, out Vector2 o, out Vector2 v)
    {
        int n = u.Count; float mu = u.Average(); Vector2 mp = Vector2.zero;
        foreach (var q in p) mp += q; mp /= n;
        float suu = 0; Vector2 sup = Vector2.zero;
        for (int i = 0; i < n; i++) { suu += (u[i] - mu) * (u[i] - mu); sup += (u[i] - mu) * (p[i] - mp); }
        v = sup / Mathf.Max(suu, 1e-6f); o = mp - mu * v;
    }

    static float Cost(Vector2[][] c, float[] u, Vector2 o, Vector2 v, float cap)
    {
        float s = 0;
        for (int i = 0; i < u.Length; i++) { Nearest(c[i], o + u[i] * v, v, out var d); d = Mathf.Min(d, cap); s += d * d; }
        return s / u.Length;
    }

    static void Refit(Vector2[][] c, float[] u, ref Vector2 o, ref Vector2 v, float keep)
    {
        var uu = new List<float>(); var pp = new List<Vector2>();
        for (int i = 0; i < u.Length; i++)
        {
            int j = Nearest(c[i], o + u[i] * v, v, out var d);
            if (d < keep) { uu.Add(u[i]); pp.Add(c[i][j]); }
        }
        if (uu.Count >= 4 && uu.Max() - uu.Min() >= 1) FitLine(uu, pp, out o, out v);
    }

    static float Pct(float[] a, float q) => a[Mathf.Clamp(Mathf.RoundToInt(q * (a.Length - 1)), 0, a.Length - 1)];

    bool Fit(out string why)
    {
        why = "";
        var on = Onsets();
        if (on.Count < 4) { why = "fewer than 4 notes in the clip"; return false; }
        int variant = Mathf.Clamp(fitVariant, 0, player.VariantCount - 1);
        var c = on.Select(x => Tips(x.frame, variant).Select(t => new Vector2(t.x, t.y)).ToArray()).ToArray();
        var u = on.Select(x => KeyCoord(x.pitch)).ToArray();
        if (u.Max() - u.Min() < 1) { why = "all notes on one key"; return false; }

        var all = c.SelectMany(a => a).ToArray();
        Vector2 mean = Vector2.zero; foreach (var p in all) mean += p; mean /= all.Length;
        float sxx = 0, sxy = 0, syy = 0;
        foreach (var p in all) { var d = p - mean; sxx += d.x * d.x; sxy += d.x * d.y; syy += d.y * d.y; }
        float ang = 0.5f * Mathf.Atan2(2 * sxy, sxx - syy);
        var axis = new Vector2(Mathf.Cos(ang), Mathf.Sin(ang));
        var s = all.Select(p => Vector2.Dot(p - mean, axis)).OrderBy(x => x).ToArray();
        var us = u.OrderBy(x => x).ToArray();
        float scale = (Pct(s, 0.9f) - Pct(s, 0.1f)) / Mathf.Max(Pct(us, 0.9f) - Pct(us, 0.1f), 1f);
        float um = Pct(us, 0.5f);

        float bestCost = float.MaxValue; Vector2 bo = Vector2.zero, bv = Vector2.right;
        foreach (float sign in new[] { 1f, -1f })
        {
            Vector2 v = axis * scale * sign, o = mean - um * v;
            for (int it = 0; it < 15; it++) Refit(c, u, ref o, ref v, it >= 2 ? 2f : float.MaxValue);
            // fingers are ~1 key apart, so polish shift/scale under a tight robust cost
            Vector2 po = o, pv = v; float pc = Cost(c, u, o, v, 0.5f);
            for (float sc = 0.9f; sc <= 1.1001f; sc += 0.025f)
                for (float sh = -1f; sh <= 1.0001f; sh += 0.05f)
                {
                    Vector2 v2 = v * sc, o2 = o + um * (v - v2) + sh * v;
                    float cc = Cost(c, u, o2, v2, 0.5f);
                    if (cc < pc) { pc = cc; po = o2; pv = v2; }
                }
            for (int k = 0; k < 3; k++) Refit(c, u, ref po, ref pv, 0.5f);
            float cost = Cost(c, u, po, pv, 2f);
            if (cost < bestCost) { bestCost = cost; bo = po; bv = pv; }
        }
        _o = bo; _v = bv;
        // key surface at the median depth of the fingertips that played
        var zs = new List<float>();
        for (int i = 0; i < u.Length; i++)
        {
            int j = Nearest(c[i], _o + u[i] * _v, _v, out var d);
            if (d < 0.5f) zs.Add(Tips(on[i].frame, variant)[j].z);
        }
        if (zs.Count == 0) { why = "no fingertip lands on a key"; return false; }
        zs.Sort();
        _depth = zs[zs.Count / 2] + 0.25f * _v.magnitude;   // tip vertex is the nail, key is a bit below
        _status = $"keyboard fitted on {player.GetVariantName(variant)}: {zs.Count}/{u.Length} onsets on a key, " +
                  $"key width {_v.magnitude * 1000:F1} mm (real 23.5)";
        return true;
    }

    // ------------------------------------------------------------ build + play

    void Build()
    {
        _built = true;
        bool fitted = false;
        if (autoFit && HasMano) fitted = Fit(out var why) || LogFail(why);
        else if (autoFit) _status = "not MANO hands: manual placement";
        if (!fitted)
        {
            _o = new Vector2(manualMiddleC.x, manualMiddleC.y); _v = new Vector2(manualStep.x, manualStep.y);
            _depth = manualMiddleC.z;
        }

        float w = _v.magnitude;
        var along = new Vector3(_v.x, _v.y, 0) / w;
        // keys extend away from the player: from the hands' centre toward the fingertips
        var n = new Vector3(-along.y, along.x, 0);
        if (HasMano)
        {
            Vector3 hc = Vector3.zero, tc = Vector3.zero;
            foreach (bool r in new[] { true, false })
            {
                var vs = player.GetVertices(r, 0, Mathf.Clamp(fitVariant, 0, player.VariantCount - 1));
                foreach (var p in vs) hc += p / vs.Length;
            }
            foreach (var t in Tips(0, Mathf.Clamp(fitVariant, 0, player.VariantCount - 1))) tc += t / 10f;
            if (Vector3.Dot(n, tc - hc * 0.5f) < 0) n = -n;
        }
        var rot = Quaternion.LookRotation(n, Vector3.back);   // local z = key length, local y = toward camera

        var root = new GameObject("MidiKeyboard").transform;
        root.SetParent(player.transform, false);
        int lo = lowNote, hi = highNote;
        if (_notes.Length > 0) { lo = Mathf.Max(lowNote, _notes.Min(x => x.pitch) - 12); hi = Mathf.Min(highNote, _notes.Max(x => x.pitch) + 12); }
        for (int m = lo; m <= hi; m++)
        {
            bool black = IsBlack(m);
            // white key: 1.6 widths in front of the touch line to 4.8 behind; black: 0.2 .. 4.8, raised
            float f0 = black ? -0.2f : -1.6f, f1 = 4.8f;
            float thick = black ? 0.9f : 0.6f;
            Vector2 c2 = _o + KeyCoord(m) * _v;
            var centre = new Vector3(c2.x, c2.y, _depth) + n * (w * (f0 + f1) / 2)
                         + Vector3.forward * (w * (black ? thick / 2 - 0.4f : thick / 2));
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Destroy(cube.GetComponent<Collider>());
            cube.name = (black ? "Black_" : "White_") + m;
            cube.transform.SetParent(root, false);
            cube.transform.localRotation = rot;
            cube.transform.localPosition = centre;
            cube.transform.localScale = new Vector3(w * (black ? 0.55f : 0.94f), w * thick, w * (f1 - f0));
            var rend = cube.GetComponent<Renderer>();
            rend.sharedMaterial = black ? _black : _white;
            _keys[m] = new Key { tr = cube.transform, restPos = centre, rend = rend, black = black };
        }
        ComputeHitRates();
    }

    bool LogFail(string why) { _status = "keyboard fit failed (" + why + "): manual placement"; Debug.LogWarning("[MidiKeyboard] " + _status); return false; }

    void ComputeHitRates()
    {
        _hitText.Clear();
        if (!HasMano) return;
        var on = Onsets();
        if (on.Count == 0) return;
        for (int vi = 0; vi < player.VariantCount; vi++)
        {
            int hits = 0; float sum = 0;
            foreach (var o in on)
            {
                var c = Tips(o.frame, vi).Select(t => new Vector2(t.x, t.y)).ToArray();
                Nearest(c, _o + KeyCoord(o.pitch) * _v, _v, out var d);
                if (d < tolerance) hits++;
                sum += d;
            }
            _hitText.Add($"{player.GetVariantName(vi)}: {100f * hits / on.Count:F0}% on key, mean {sum / on.Count * 23.5f:F0} mm");
        }
    }

    void Update()
    {
        if (player == null || _notes == null || player.Fps <= 0 || player.FrameCount == 0) return;
        if (!_built) Build();
        float t = player.CurrentFrame / player.Fps + timeOffset;
        var active = new Dictionary<int, Material>();
        Vector2[] tips = null; bool[] isRight = null;
        if (HasMano)
        {
            var tp = Tips(player.CurrentFrame, -1);
            tips = tp.Select(x => new Vector2(x.x, x.y)).ToArray();
            isRight = Enumerable.Range(0, 10).Select(i => i < 5).ToArray();
        }
        foreach (var nt in _notes)
        {
            if (t < nt.on || t >= nt.off || !_keys.ContainsKey(nt.pitch)) continue;
            Material mat = _right;
            if (tips != null)
            {
                int j = Nearest(tips, _o + KeyCoord(nt.pitch) * _v, _v, out var d);
                mat = d > tolerance ? _miss : isRight[j] ? _right : _left;
            }
            active[nt.pitch] = mat;
        }
        float travel = 0.35f * _v.magnitude;
        foreach (var kv in _keys)
        {
            bool down = active.TryGetValue(kv.Key, out var mat);
            kv.Value.rend.sharedMaterial = down ? mat : (kv.Value.black ? _black : _white);
            kv.Value.tr.localPosition = kv.Value.restPos + (down ? Vector3.forward * travel : Vector3.zero);
        }
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 52, 900, 20), _status);
        for (int i = 0; i < _hitText.Count; i++)
            GUI.Label(new Rect(10, 72 + 18 * i, 900, 20), (i == player.Variant ? "> " : "  ") + _hitText[i]);
    }
}
