// Builds a piano keyboard (C2..C6) in front of the hands and presses keys in sync with
// HandMotionPlayer. Layout and notes come from demo_notes.json (tools/bake_demo_hands.js),
// so keys line up with the baked hand motion.
//
// Setup: add to any GameObject, drag the HandMotionPlayer into Player, keep notes file name.
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class PianoKeyboard : MonoBehaviour
{
    public HandMotionPlayer player;
    [Tooltip("File inside Assets/StreamingAssets")]
    public string notesFile = "demo_notes.json";
    public Color whiteColor = new Color(0.97f, 0.96f, 0.93f);
    public Color blackColor = new Color(0.09f, 0.09f, 0.11f);
    public Color pressedColor = new Color(0.85f, 0.55f, 0.17f);
    public Color caseColor = new Color(0.11f, 0.11f, 0.13f);

    [System.Serializable] class Note { public int pitch; public float on; public float off; public string hand; public int finger; }
    [System.Serializable] class Layout { public float scale; public float x0; public int low; public int high; public float keyTravel; public float[] center; public Note[] notes; }

    class Key { public Transform pivot; public Renderer rend; public bool black; public float length; }

    Layout _l;
    readonly Dictionary<int, Key> _keys = new Dictionary<int, Key>();
    Material _white, _black, _pressed;

    static bool IsBlack(int m) { int k = m % 12; return k == 1 || k == 3 || k == 6 || k == 8 || k == 10; }

    void Start()
    {
        _l = JsonUtility.FromJson<Layout>(File.ReadAllText(Path.Combine(Application.streamingAssetsPath, notesFile)));
        var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
        _white = new Material(shader) { color = whiteColor };
        _black = new Material(shader) { color = blackColor };
        _pressed = new Material(shader) { color = pressedColor };

        var whiteIndex = new Dictionary<int, int>();
        int wi = 0;
        for (int m = _l.low; m <= _l.high; m++) if (!IsBlack(m)) whiteIndex[m] = wi++;

        for (int m = _l.low; m <= _l.high; m++)
        {
            bool black = IsBlack(m);
            float x = black ? _l.x0 + whiteIndex[m - 1] + 0.5f : _l.x0 + whiteIndex[m];
            // demo units: white keys z -3..3, black z -3..0.9; top surface y=0 (white), 0.45 (black)
            float len = black ? 3.9f : 6f, w = black ? 0.58f : 0.96f, h = black ? 0.55f : 0.6f, top = black ? 0.45f : 0f;
            var pivot = new GameObject((black ? "Black_" : "White_") + m).transform;
            pivot.SetParent(transform, false);
            pivot.localPosition = ToUnity(x, top, -3f);            // hinge at the back edge
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            Destroy(cube.GetComponent<Collider>());
            cube.transform.SetParent(pivot, false);
            cube.transform.localScale = new Vector3(w, h, len) * _l.scale;
            cube.transform.localPosition = new Vector3(0, -h / 2, -len / 2) * _l.scale;   // extends toward the player (-z)
            var r = cube.GetComponent<Renderer>();
            r.sharedMaterial = black ? _black : _white;
            _keys[m] = new Key { pivot = pivot, rend = r, black = black, length = len };
        }
        // case under the keys
        var body = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Destroy(body.GetComponent<Collider>());
        body.name = "Case";
        body.transform.SetParent(transform, false);
        body.transform.localScale = new Vector3(wi + 1.2f, 1.4f, 8.5f) * _l.scale;
        body.transform.localPosition = ToUnity(_l.x0 + (wi - 1) / 2f, -1.35f, -0.9f);
        body.GetComponent<Renderer>().sharedMaterial = new Material(shader) { color = caseColor };
    }

    Vector3 ToUnity(float x, float y, float z) =>
        new Vector3((x - _l.center[0]) * _l.scale, (y - _l.center[1]) * _l.scale, -(z - _l.center[2]) * _l.scale);

    static float Ease(float x) => x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x);

    static float PressAt(Note n, float t)
    {
        const float A = 0.09f, R = 0.1f;
        if (t < n.on - A || t > n.off + R) return 0;
        if (t < n.on) return Ease((t - (n.on - A)) / A);
        if (t <= n.off) return 1;
        return 1 - Ease((t - n.off) / R);
    }

    void Update()
    {
        if (_l == null || player == null || player.Fps <= 0) return;
        float t = player.CurrentFrame / player.Fps;
        var down = new Dictionary<int, float>();
        foreach (var n in _l.notes)
        {
            float p = PressAt(n, t);
            if (p <= 0) continue;
            float d = Mathf.Max(0, (p - 0.55f) / 0.45f);
            down[n.pitch] = down.TryGetValue(n.pitch, out var o) ? Mathf.Max(o, d) : d;
        }
        foreach (var kv in _keys)
        {
            float d = down.TryGetValue(kv.Key, out var v) ? v : 0;
            float ang = Mathf.Atan2(d * _l.keyTravel, kv.Value.length) * Mathf.Rad2Deg;
            kv.Value.pivot.localRotation = Quaternion.Euler(-ang, 0, 0);   // front edge goes down
            kv.Value.rend.sharedMaterial = d > 0.05f ? _pressed : (kv.Value.black ? _black : _white);
        }
    }
}
