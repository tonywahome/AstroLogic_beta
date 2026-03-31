using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.UI;

/// <summary>
/// Editor script to programmatically build the AstroExploration scene.
/// Use via menu: AstroLogic > Setup Scene
/// </summary>
public class SceneSetup : EditorWindow
{
    private static float distanceScale = 10f; // World units per AU
    private static float bodyScaleMultiplier = 100f; // Visual exaggeration

    [MenuItem("AstroLogic/Setup Scene")]
    public static void SetupScene()
    {
        // Create new scene
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // Remove default directional light
        var defaultLight = GameObject.Find("Directional Light");
        if (defaultLight != null) DestroyImmediate(defaultLight);

        // Create managers
        CreateManagers();

        // Create celestial bodies
        CreateSun();
        CreatePlanet("Earth", 1.0f, 365.25f, 0f, new Color(0.27f, 0.51f, 0.78f), 0.3f, new string[] { }, 0f);
        CreatePlanet("Mars", 1.524f, 687f, 45f, new Color(0.76f, 0.27f, 0.05f), 0.25f, new string[] { "ice", "organic_compounds" }, 0.5f);
        CreatePlanet("Jupiter", 5.203f, 4332.59f, 60f, new Color(0.79f, 0.69f, 0.51f), 0.8f, new string[] { }, 0f);
        CreatePlanet("Saturn", 9.537f, 10759.22f, 120f, new Color(0.82f, 0.71f, 0.39f), 0.7f, new string[] { }, 0f);

        // Create moons
        CreateMoon("Europa", "Jupiter", 0.045f, 3.55f, 0f, new Color(0.78f, 0.86f, 0.94f), 0.12f,
            new string[] { "liquid_water", "organic_compounds" }, 0.3f);
        CreateMoon("Enceladus", "Saturn", 0.016f, 1.37f, 90f, new Color(0.71f, 0.78f, 0.86f), 0.1f,
            new string[] { "liquid_water", "ice", "signs_of_intelligence" }, 0.2f);

        // Create spacecraft
        CreateSpacecraft();

        // Create UI canvas
        CreateUICanvas();

        // Create star background
        CreateStarfield();

        // Set camera
        SetupCamera();

        // Configure physics
        Physics.gravity = Vector3.zero;

        // Save scene
        string scenePath = "Assets/Scenes/AstroExploration.unity";
        EditorSceneManager.SaveScene(scene, scenePath);
        Debug.Log($"AstroExploration scene created at {scenePath}");
    }

    private static void CreateManagers()
    {
        // Solar System Manager
        GameObject ssmObj = new GameObject("SolarSystemManager");
        var ssm = ssmObj.AddComponent<SolarSystemManager>();
        ssm.distanceScale = distanceScale;
        ssm.bodyScaleMultiplier = bodyScaleMultiplier;

        // Mission Manager
        GameObject mmObj = new GameObject("MissionManager");
        mmObj.AddComponent<MissionManager>();
    }

    private static void CreateSun()
    {
        GameObject sun = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sun.name = "Sun";
        sun.transform.position = Vector3.zero;
        sun.transform.localScale = Vector3.one * 1.5f;

        // Emissive material
        var rend = sun.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(1f, 0.87f, 0f);
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", new Color(1f, 0.87f, 0f) * 3f);
        rend.material = mat;

        // Point light
        var light = sun.AddComponent<Light>();
        light.type = LightType.Point;
        light.color = new Color(1f, 0.95f, 0.8f);
        light.intensity = 2f;
        light.range = 200f;

        // CelestialBody component
        var cb = sun.AddComponent<CelestialBody>();
        cb.mass = 1.0f;
        cb.bodyRadius = 0.75f;
        cb.orbitRadius = 0f;
        cb.orbitPeriod = 0f;
        cb.bodyColor = new Color(1f, 0.87f, 0f);

        // Register with manager
        var ssm = FindObjectOfType<SolarSystemManager>();
        if (ssm != null) ssm.RegisterBody("Sun", cb);
    }

    private static void CreatePlanet(string name, float orbitRadiusAU, float periodDays,
        float initialAngleDeg, Color color, float visualScale, string[] biosigs, float detectionZone)
    {
        GameObject planet = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        planet.name = name;

        float worldOrbitRadius = orbitRadiusAU * distanceScale;
        float angle = initialAngleDeg * Mathf.Deg2Rad;
        planet.transform.position = new Vector3(
            worldOrbitRadius * Mathf.Cos(angle),
            0f,
            worldOrbitRadius * Mathf.Sin(angle)
        );
        planet.transform.localScale = Vector3.one * visualScale;

        // Material
        var rend = planet.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        rend.material = mat;

        // CelestialBody component
        var cb = planet.AddComponent<CelestialBody>();
        cb.mass = GetBodyMass(name);
        cb.bodyRadius = visualScale * 0.5f;
        cb.orbitRadius = worldOrbitRadius;
        cb.orbitPeriod = periodDays;
        cb.initialAngle = angle;
        cb.bodyColor = color;
        cb.biosignatures = biosigs;
        cb.detectionZoneRadius = detectionZone * distanceScale;

        // Register
        var ssm = FindObjectOfType<SolarSystemManager>();
        if (ssm != null) ssm.RegisterBody(name, cb);

        // Create biosignature zone if applicable
        if (detectionZone > 0 && biosigs.Length > 0)
        {
            foreach (string biosig in biosigs)
            {
                CreateBiosignatureZone(planet, cb, biosig, detectionZone * distanceScale);
            }
        }

        // Orbital path visualization
        CreateOrbitalPath(worldOrbitRadius, color);
    }

    private static void CreateMoon(string name, string parentName, float orbitRadiusAU,
        float periodDays, float initialAngleDeg, Color color, float visualScale,
        string[] biosigs, float detectionZone)
    {
        GameObject parent = GameObject.Find(parentName);
        if (parent == null) return;

        GameObject moon = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        moon.name = name;

        float worldOrbitRadius = orbitRadiusAU * distanceScale;
        float angle = initialAngleDeg * Mathf.Deg2Rad;
        moon.transform.position = parent.transform.position + new Vector3(
            worldOrbitRadius * Mathf.Cos(angle),
            0f,
            worldOrbitRadius * Mathf.Sin(angle)
        );
        moon.transform.localScale = Vector3.one * visualScale;

        // Material
        var rend = moon.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        rend.material = mat;

        // CelestialBody component
        var cb = moon.AddComponent<CelestialBody>();
        cb.mass = GetBodyMass(name);
        cb.bodyRadius = visualScale * 0.5f;
        cb.orbitRadius = worldOrbitRadius;
        cb.orbitPeriod = periodDays;
        cb.initialAngle = angle;
        cb.parentBody = parent.transform;
        cb.bodyColor = color;
        cb.biosignatures = biosigs;
        cb.detectionZoneRadius = detectionZone * distanceScale;

        // Register
        var ssm = FindObjectOfType<SolarSystemManager>();
        if (ssm != null) ssm.RegisterBody(name, cb);

        // Biosignature zones
        if (detectionZone > 0 && biosigs.Length > 0)
        {
            foreach (string biosig in biosigs)
            {
                CreateBiosignatureZone(moon, cb, biosig, detectionZone * distanceScale);
            }
        }
    }

    private static void CreateBiosignatureZone(GameObject parent, CelestialBody body,
        string biosigType, float radius)
    {
        GameObject zone = new GameObject($"BiosigZone_{biosigType}");
        zone.transform.SetParent(parent.transform);
        zone.transform.localPosition = Vector3.zero;

        var bz = zone.AddComponent<BiosignatureZone>();
        bz.biosignatureType = biosigType;
        bz.detectionRadius = radius;
        bz.parentBody = body;

        // Color based on type
        switch (biosigType)
        {
            case "liquid_water":
                bz.zoneColor = new Color(0f, 0.4f, 1f, 0.12f); break;
            case "ice":
                bz.zoneColor = new Color(0.4f, 0.8f, 1f, 0.12f); break;
            case "organic_compounds":
                bz.zoneColor = new Color(0f, 0.8f, 0.2f, 0.12f); break;
            case "signs_of_intelligence":
                bz.zoneColor = new Color(0.8f, 0f, 1f, 0.12f); break;
        }
    }

    private static void CreateSpacecraft()
    {
        GameObject spacecraft = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        spacecraft.name = "Spacecraft";
        spacecraft.tag = "Spacecraft";
        spacecraft.transform.position = new Vector3(distanceScale, 0, 0); // Earth orbit
        spacecraft.transform.localScale = new Vector3(0.15f, 0.3f, 0.15f);

        // Material
        var rend = spacecraft.GetComponent<Renderer>();
        var mat = new Material(Shader.Find("Standard"));
        mat.color = Color.white;
        mat.EnableKeyword("_EMISSION");
        mat.SetColor("_EmissionColor", Color.white * 0.5f);
        rend.material = mat;

        // Rigidbody
        var rb = spacecraft.AddComponent<Rigidbody>();
        rb.useGravity = false;
        rb.mass = 1f;
        rb.linearDamping = 0f;
        rb.angularDamping = 0f;

        // Trail renderer for trajectory
        var trail = spacecraft.AddComponent<TrailRenderer>();
        trail.time = 30f;
        trail.startWidth = 0.05f;
        trail.endWidth = 0.01f;
        trail.material = new Material(Shader.Find("Sprites/Default"));
        trail.startColor = new Color(1f, 1f, 1f, 0.8f);
        trail.endColor = new Color(1f, 1f, 1f, 0.1f);

        // Thrust particle system
        GameObject thrustFX = new GameObject("ThrustFX");
        thrustFX.transform.SetParent(spacecraft.transform);
        thrustFX.transform.localPosition = new Vector3(0, -0.3f, 0);
        thrustFX.transform.localRotation = Quaternion.Euler(90, 0, 0);
        var ps = thrustFX.AddComponent<ParticleSystem>();
        var main = ps.main;
        main.startColor = new Color(1f, 0.5f, 0f);
        main.startSize = 0.1f;
        main.startLifetime = 0.5f;
        main.maxParticles = 50;
        var emission = ps.emission;
        emission.rateOverTime = 0;

        // ML-Agents components
        var agent = spacecraft.AddComponent<SpacecraftAgent>();

        // Configure BehaviorParameters (added automatically with Agent)
        var bp = spacecraft.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        if (bp != null)
        {
            bp.BrainParameters.VectorObservationSize = 23;
            bp.BrainParameters.ActionSpec = new Unity.MLAgents.Actuators.ActionSpec(
                0, new int[] { 5, 3, 3, 3, 4, 2 }
            );
            bp.BehaviorName = "SpacecraftAgent";
        }

        // Decision Requester
        var dr = spacecraft.AddComponent<Unity.MLAgents.DecisionRequester>();
        dr.DecisionPeriod = 1;
    }

    private static void CreateOrbitalPath(float radius, Color color)
    {
        GameObject path = new GameObject($"OrbitPath_{radius:F0}");
        var lr = path.AddComponent<LineRenderer>();
        lr.useWorldSpace = true;
        lr.startWidth = 0.02f;
        lr.endWidth = 0.02f;
        lr.material = new Material(Shader.Find("Sprites/Default"));

        Color dimColor = new Color(color.r * 0.3f, color.g * 0.3f, color.b * 0.3f, 0.4f);
        lr.startColor = dimColor;
        lr.endColor = dimColor;

        int segments = 128;
        lr.positionCount = segments + 1;
        for (int i = 0; i <= segments; i++)
        {
            float angle = 2f * Mathf.PI * i / segments;
            lr.SetPosition(i, new Vector3(
                radius * Mathf.Cos(angle),
                0f,
                radius * Mathf.Sin(angle)
            ));
        }
    }

    private static void CreateUICanvas()
    {
        GameObject canvasObj = new GameObject("HUDCanvas");
        var canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvasObj.AddComponent<CanvasScaler>();
        canvasObj.AddComponent<GraphicRaycaster>();

        var uiManager = canvasObj.AddComponent<UIManager>();

        // Fuel bar
        uiManager.fuelText = CreateUIText(canvasObj.transform, "FuelText", "FUEL: 100.0%",
            new Vector2(20, -20), TextAnchor.UpperLeft);
        // Battery bar
        uiManager.batteryText = CreateUIText(canvasObj.transform, "BatteryText", "BATT: 100.0%",
            new Vector2(20, -45), TextAnchor.UpperLeft);
        // Instrument
        uiManager.instrumentText = CreateUIText(canvasObj.transform, "InstrumentText", "INSTRUMENT: None",
            new Vector2(-20, -20), TextAnchor.UpperRight);
        // Comm status
        uiManager.commStatusText = CreateUIText(canvasObj.transform, "CommText", "IDLE",
            new Vector2(-20, -45), TextAnchor.UpperRight);
        // Biosig
        uiManager.biosigText = CreateUIText(canvasObj.transform, "BiosigText", "FOUND: 0  TX: 0",
            new Vector2(-20, -70), TextAnchor.UpperRight);
        // Step counter
        uiManager.stepCounterText = CreateUIText(canvasObj.transform, "StepText", "Step: 0 / 100,000",
            new Vector2(0, 20), TextAnchor.LowerCenter);
        // Reward
        uiManager.rewardText = CreateUIText(canvasObj.transform, "RewardText", "Reward: 0.00",
            new Vector2(0, 45), TextAnchor.LowerCenter);
        // Position
        uiManager.positionText = CreateUIText(canvasObj.transform, "PosText", "POS: (0, 0, 0)",
            new Vector2(-20, 20), TextAnchor.LowerRight);
        // Velocity
        uiManager.velocityText = CreateUIText(canvasObj.transform, "VelText", "VEL: 0.0000",
            new Vector2(-20, 45), TextAnchor.LowerRight);
        // SNR
        uiManager.snrText = CreateUIText(canvasObj.transform, "SNRText", "SNR: 0.000",
            new Vector2(-20, 70), TextAnchor.LowerRight);
    }

    private static Text CreateUIText(Transform parent, string name, string defaultText,
        Vector2 offset, TextAnchor anchor)
    {
        GameObject textObj = new GameObject(name);
        textObj.transform.SetParent(parent);

        var rect = textObj.AddComponent<RectTransform>();

        // Position based on anchor
        switch (anchor)
        {
            case TextAnchor.UpperLeft:
                rect.anchorMin = new Vector2(0, 1);
                rect.anchorMax = new Vector2(0, 1);
                rect.pivot = new Vector2(0, 1);
                break;
            case TextAnchor.UpperRight:
                rect.anchorMin = new Vector2(1, 1);
                rect.anchorMax = new Vector2(1, 1);
                rect.pivot = new Vector2(1, 1);
                break;
            case TextAnchor.LowerCenter:
                rect.anchorMin = new Vector2(0.5f, 0);
                rect.anchorMax = new Vector2(0.5f, 0);
                rect.pivot = new Vector2(0.5f, 0);
                break;
            case TextAnchor.LowerRight:
                rect.anchorMin = new Vector2(1, 0);
                rect.anchorMax = new Vector2(1, 0);
                rect.pivot = new Vector2(1, 0);
                break;
        }

        rect.anchoredPosition = offset;
        rect.sizeDelta = new Vector2(300, 25);

        var text = textObj.AddComponent<Text>();
        text.text = defaultText;
        text.font = Font.CreateDynamicFontFromOSFont("Consolas", 14);
        text.fontSize = 14;
        text.color = new Color(0.9f, 0.9f, 0.95f);
        text.alignment = anchor;

        return text;
    }

    private static void CreateStarfield()
    {
        GameObject starfield = new GameObject("Starfield");
        var ps = starfield.AddComponent<ParticleSystem>();

        var main = ps.main;
        main.maxParticles = 500;
        main.startLifetime = float.MaxValue;
        main.startSpeed = 0;
        main.startSize = 0.3f;
        main.startColor = new Color(0.9f, 0.9f, 1f);
        main.simulationSpace = ParticleSystemSimulationSpace.World;
        main.playOnAwake = true;

        var shape = ps.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = 150f;

        var emission = ps.emission;
        emission.rateOverTime = 0;
        var burst = new ParticleSystem.Burst(0f, 500);
        emission.SetBursts(new ParticleSystem.Burst[] { burst });

        var rend = ps.GetComponent<ParticleSystemRenderer>();
        rend.material = new Material(Shader.Find("Particles/Standard Unlit"));
    }

    private static void SetupCamera()
    {
        var mainCam = Camera.main;
        if (mainCam != null)
        {
            mainCam.transform.position = new Vector3(0, 80, 0);
            mainCam.transform.rotation = Quaternion.Euler(90, 0, 0);
            mainCam.clearFlags = CameraClearFlags.SolidColor;
            mainCam.backgroundColor = new Color(0.02f, 0.02f, 0.06f);
            mainCam.farClipPlane = 500f;

            // Add simple follow script
            var follow = mainCam.gameObject.AddComponent<CameraFollow>();
        }
    }

    private static float GetBodyMass(string name)
    {
        switch (name)
        {
            case "Sun": return 1.0f;
            case "Earth": return 3.003e-6f;
            case "Mars": return 3.213e-7f;
            case "Jupiter": return 9.543e-4f;
            case "Europa": return 2.528e-8f;
            case "Saturn": return 2.857e-4f;
            case "Enceladus": return 1.08e-10f;
            default: return 1e-7f;
        }
    }
}

/// <summary>
/// Simple camera follow script that tracks the spacecraft from above.
/// </summary>
public class CameraFollow : MonoBehaviour
{
    public float height = 80f;
    public float smoothSpeed = 2f;
    private Transform target;

    void Start()
    {
        var spacecraft = GameObject.FindWithTag("Spacecraft");
        if (spacecraft != null)
            target = spacecraft.transform;
    }

    void LateUpdate()
    {
        if (target == null) return;

        Vector3 desiredPos = new Vector3(target.position.x, height, target.position.z);
        transform.position = Vector3.Lerp(transform.position, desiredPos, smoothSpeed * Time.deltaTime);
        transform.LookAt(target.position);
    }
}
