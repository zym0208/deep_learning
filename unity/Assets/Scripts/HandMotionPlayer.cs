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
// Week 3, version switching: fill Variant Files with e.g. raw.hmotion, smooth.hmotion,
// refined.hmotion (same clip, same frame count, from tools/export_unity.py). Keys 1/2/3
// (or Tab to cycle) switch version on the same frame without restarting audio, so raw and
// refined can be compared side by side in time. Variant Files overrides File Name.
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
    [Tooltip("Optional: several versions of the same clip (raw / smoothed / refined). Overrides File Name.")]
    public string[] variantFiles = new string[0];
    [Tooltip("Seconds added to the audio time before picking a frame; positive = hands earlier.")]
    public float audioOffset = 0f;
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

    public int Variant { get; private set; }
    public string VariantName => _names.Length > 0 ? _names[Variant] : fileName;
    public int VariantCount => _varR != null ? _varR.Length : 1;
    public string GetVariantName(int i) => _names.Length > 0 ? _names[i] : fileName;

    // Vertices of one hand at a frame, in this object's local space (read-only, do not modify).
    // variant -1 = the version on screen. Used by MidiKeyboard to find the fingertips.
    public Vector3[] GetVertices(bool right, int frame, int variant = -1)
    {
        var frames = right ? _framesR : _framesL;
        if (variant >= 0 && _varR != null && variant < _varR.Length) frames = right ? _varR[variant] : _varL[variant];
        return frames == null ? null : frames[Mathf.Clamp(frame, 0, FrameCount - 1)];
    }

    Mesh _right, _left;
    Vector3[][] _framesR, _framesL;
    Vector3[][][] _varR, _varL;
    string[] _names = new string[0];
    float _clock;   // seconds, used when there is no audio

    void Start()
    {
        if (variantFiles != null && variantFiles.Length > 0) LoadVariants(variantFiles);
        else Load(Path.Combine(Application.streamingAssetsPath, fileName));
        if (autoCamera && Camera.main != null)
        {
            Camera.main.transform.position = transform.position + new Vector3(0, 0, -cameraDistance);
            Camera.main.transform.LookAt(transform.position);
            Camera.main.nearClipPlane = 0.01f;
        }
        ShowFrame(0);
        if (playOnStart) Play();
    }

    // Load several versions of one clip; the meshes are built from the first, the others only
    // supply vertex positions, so all must share vertex counts. Frame count = shortest file.
    public void LoadVariants(string[] files)
    {
        _names = files;
        _varR = new Vector3[files.Length][][];
        _varL = new Vector3[files.Length][][];
        int frames = int.MaxValue, nR = -1, nL = -1;
        float fps = 0;
        for (int i = 0; i < files.Length; i++)
        {
            Load(Path.Combine(Application.streamingAssetsPath, files[i]), i == 0);
            if (i == 0) { nR = _framesR[0].Length; nL = _framesL[0].Length; fps = Fps; }
            else if (_framesR[0].Length != nR || _framesL[0].Length != nL)
                throw new InvalidDataException($"{files[i]}: vertex count differs from {files[0]}");
            else if (Mathf.Abs(Fps - fps) > 1e-3f)
                Debug.LogWarning($"[HandMotionPlayer] {files[i]} is {Fps} fps, {files[0]} is {fps} fps; using {fps}");
            _varR[i] = _framesR; _varL[i] = _framesL;
            frames = Mathf.Min(frames, FrameCount);
        }
        Fps = fps; FrameCount = frames;
        SetVariant(0);
    }

    public void SetVariant(int i)
    {
        if (_varR == null || i < 0 || i >= _varR.Length) return;
        Variant = i;
        _framesR = _varR[i]; _framesL = _varL[i];
        if (_right != null) ShowFrame(CurrentFrame);
    }

    public void Load(string path) => Load(path, true);

    void Load(string path, bool buildMeshes)
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
            if (buildMeshes)
            {
                _right = MakeHand("RightHand", _framesR[0], trisR);
                _left = MakeHand("LeftHand", _framesL[0], trisL);
            }
        }
        Debug.Log($"[HandMotionPlayer] loaded {Path.GetFileName(path)}: {FrameCount} frames @ {Fps} fps");
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
        if (audioSource != null && audioSource.clip != null && _clock < audioSource.clip.length)
        {
            audioSource.time = Mathf.Max(0f, _clock);   // resume where the hands are
            audioSource.Play();
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

    // Audio is the master clock while it plays. timeSamples is sample-exact (audioSource.time is
    // updated in coarse steps on some platforms, which makes the hands stutter against the sound).
    // If the audio ends before the motion, the internal clock carries on from where audio stopped.
    bool AudioRunning => audioSource != null && audioSource.clip != null && audioSource.isPlaying;
    float CurrentTime => _clock + audioOffset;

    void Update()
    {
        HandleKeys();
        if (!IsPlaying || FrameCount == 0) return;

        if (AudioRunning) _clock = (float)audioSource.timeSamples / audioSource.clip.frequency;
        else _clock += Time.deltaTime;
        int frame = Mathf.FloorToInt(Mathf.Max(0f, CurrentTime) * Fps);
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
        bool tab = kb.tabKey.wasPressedThisFrame;
        int num = kb.digit1Key.wasPressedThisFrame ? 0 : kb.digit2Key.wasPressedThisFrame ? 1
                : kb.digit3Key.wasPressedThisFrame ? 2 : kb.digit4Key.wasPressedThisFrame ? 3 : -1;
#else
        bool space = Input.GetKeyDown(KeyCode.Space), r = Input.GetKeyDown(KeyCode.R);
        bool left = Input.GetKeyDown(KeyCode.LeftArrow), right = Input.GetKeyDown(KeyCode.RightArrow);
        bool tab = Input.GetKeyDown(KeyCode.Tab);
        int num = Input.GetKeyDown(KeyCode.Alpha1) ? 0 : Input.GetKeyDown(KeyCode.Alpha2) ? 1
                : Input.GetKeyDown(KeyCode.Alpha3) ? 2 : Input.GetKeyDown(KeyCode.Alpha4) ? 3 : -1;
#endif
        if (space) { if (IsPlaying) Pause(); else Play(); }
        if (r) Restart();
        if (left) Step(-1);
        if (right) Step(1);
        if (num >= 0) SetVariant(num);
        if (tab && _names.Length > 1) SetVariant((Variant + 1) % _names.Length);
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 600, 20),
            $"{VariantName}  frame {CurrentFrame + 1}/{FrameCount}  t={CurrentFrame / Mathf.Max(Fps, 1):F2}s  " +
            (IsPlaying ? "playing" : "paused") + "   [Space] play/pause  [R] restart  [←/→] step" +
            (_names.Length > 1 ? "  [1-" + _names.Length + "/Tab] version" : ""));
        if (_names.Length > 1)
            GUI.Label(new Rect(10, 30, 600, 24), $"<size=18><b>Version {Variant + 1}: {VariantName}</b></size>");
    }
}
