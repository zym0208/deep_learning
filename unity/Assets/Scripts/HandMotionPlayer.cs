// Plays a .hmotion file (written by tools/export_unity.py): two hand meshes whose
// vertices are replaced every frame. Optional AudioSource drives the clock so the
// hands stay in sync with the music.
//
// Setup (Week 1):
//   1. Copy the .hmotion file into Assets/StreamingAssets/.
//   2. Create an empty GameObject, add this component, set File Name.
//   3. (Optional) Add an AudioSource with the clip's audio and drag it into Audio.
//   4. Press Play. Space = play/pause, R = restart, Left/Right = step one frame while paused.
//
// Works with both the legacy Input Manager and the new Input System package.
using System;
using System.IO;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public class HandMotionPlayer : MonoBehaviour
{
    [Tooltip("File inside Assets/StreamingAssets, e.g. dummy.hmotion")]
    public string fileName = "dummy.hmotion";
    public AudioSource audioSource;
    public Material handMaterial;
    public bool playOnStart = true;
    public bool loop = true;
    [Tooltip("Tick if the hands look inside-out (only the back faces render).")]
    public bool flipWinding = false;
    [Tooltip("Place the main camera at (0,0,-cameraDistance) looking at the hands.")]
    public bool autoCamera = true;
    public float cameraDistance = 1.0f;

    public float Fps { get; private set; }
    public int FrameCount { get; private set; }
    public int CurrentFrame { get; private set; }
    public bool IsPlaying { get; private set; }

    Mesh _right, _left;
    Vector3[][] _framesR, _framesL;
    float _clock;   // seconds, used when there is no audio

    void Start()
    {
        Load(Path.Combine(Application.streamingAssetsPath, fileName));
        if (autoCamera && Camera.main != null)
        {
            Camera.main.transform.position = transform.position + new Vector3(0, 0, -cameraDistance);
            Camera.main.transform.LookAt(transform.position);
            Camera.main.nearClipPlane = 0.01f;
        }
        ShowFrame(0);
        if (playOnStart) Play();
    }

    public void Load(string path)
    {
        using (var br = new BinaryReader(File.OpenRead(path)))
        {
            var magic = new string(br.ReadChars(4));
            if (magic != "HMOT") throw new InvalidDataException($"{path} is not a .hmotion file");
            int version = br.ReadInt32();
            if (version != 1) throw new InvalidDataException($"unsupported .hmotion version {version}");
            Fps = br.ReadSingle();
            FrameCount = br.ReadInt32();
            int nVertR = br.ReadInt32(), nTriR = br.ReadInt32();
            int nVertL = br.ReadInt32(), nTriL = br.ReadInt32();
            int[] trisR = ReadInts(br, nTriR * 3), trisL = ReadInts(br, nTriL * 3);
            _framesR = new Vector3[FrameCount][];
            _framesL = new Vector3[FrameCount][];
            for (int f = 0; f < FrameCount; f++)
            {
                _framesR[f] = ReadVerts(br, nVertR);
                _framesL[f] = ReadVerts(br, nVertL);
            }
            _right = MakeHand("RightHand", _framesR[0], trisR);
            _left = MakeHand("LeftHand", _framesL[0], trisL);
        }
        Debug.Log($"[HandMotionPlayer] loaded {fileName}: {FrameCount} frames @ {Fps} fps");
    }

    static int[] ReadInts(BinaryReader br, int n)
    {
        var a = new int[n];
        for (int i = 0; i < n; i++) a[i] = br.ReadInt32();
        return a;
    }

    static Vector3[] ReadVerts(BinaryReader br, int n)
    {
        var v = new Vector3[n];
        for (int i = 0; i < n; i++) v[i] = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
        return v;
    }

    Mesh MakeHand(string name, Vector3[] verts, int[] tris)
    {
        if (flipWinding)
            for (int i = 0; i < tris.Length; i += 3) (tris[i + 1], tris[i + 2]) = (tris[i + 2], tris[i + 1]);
        var go = new GameObject(name);
        go.transform.SetParent(transform, false);
        var mesh = new Mesh { name = name };
        mesh.MarkDynamic();
        mesh.vertices = verts;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        go.AddComponent<MeshFilter>().sharedMesh = mesh;
        var mr = go.AddComponent<MeshRenderer>();
        mr.sharedMaterial = handMaterial != null ? handMaterial : DefaultMaterial();
        return mesh;
    }

    static Material DefaultMaterial()
    {
        var shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
        var m = new Material(shader) { color = new Color(0.95f, 0.78f, 0.68f) };
        return m;
    }

    // ------------------------------------------------------------ playback

    public void Play()
    {
        IsPlaying = true;
        if (audioSource != null && audioSource.clip != null)
        {
            // Play() would restart from 0; UnPause() continues from audioSource.time
            if (audioSource.time > 0f) audioSource.UnPause(); else audioSource.Play();
        }
    }

    public void Pause()
    {
        IsPlaying = false;
        if (audioSource != null) audioSource.Pause();
    }

    public void Restart()
    {
        _clock = 0;
        if (audioSource != null) { audioSource.Stop(); audioSource.time = 0; }
        ShowFrame(0);
        Play();
    }

    float CurrentTime =>
        audioSource != null && audioSource.clip != null ? audioSource.time : _clock;

    void Update()
    {
        HandleKeys();
        if (!IsPlaying || FrameCount == 0) return;

        _clock += Time.deltaTime;
        int frame = Mathf.FloorToInt(CurrentTime * Fps);
        if (frame >= FrameCount)
        {
            if (loop) { Restart(); return; }
            frame = FrameCount - 1;
            Pause();
        }
        if (frame != CurrentFrame) ShowFrame(frame);
    }

    public void ShowFrame(int f)
    {
        CurrentFrame = Mathf.Clamp(f, 0, FrameCount - 1);
        _right.vertices = _framesR[CurrentFrame];
        _left.vertices = _framesL[CurrentFrame];
        _right.RecalculateNormals();
        _left.RecalculateNormals();
        _right.RecalculateBounds();
        _left.RecalculateBounds();
    }

    void Step(int d)
    {
        Pause();
        ShowFrame(CurrentFrame + d);
        _clock = CurrentFrame / Fps;
        if (audioSource != null && audioSource.clip != null)
            audioSource.time = Mathf.Min(_clock, audioSource.clip.length - 0.001f);
    }

    void HandleKeys()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (kb == null) return;
        bool space = kb.spaceKey.wasPressedThisFrame, r = kb.rKey.wasPressedThisFrame;
        bool left = kb.leftArrowKey.wasPressedThisFrame, right = kb.rightArrowKey.wasPressedThisFrame;
#else
        bool space = Input.GetKeyDown(KeyCode.Space), r = Input.GetKeyDown(KeyCode.R);
        bool left = Input.GetKeyDown(KeyCode.LeftArrow), right = Input.GetKeyDown(KeyCode.RightArrow);
#endif
        if (space) { if (IsPlaying) Pause(); else Play(); }
        if (r) Restart();
        if (left) Step(-1);
        if (right) Step(1);
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 600, 20),
            $"{fileName}  frame {CurrentFrame + 1}/{FrameCount}  t={CurrentFrame / Mathf.Max(Fps, 1):F2}s  " +
            (IsPlaying ? "playing" : "paused") + "   [Space] play/pause  [R] restart  [←/→] step");
    }
}
