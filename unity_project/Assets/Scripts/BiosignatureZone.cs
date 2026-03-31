using UnityEngine;

/// <summary>
/// Trigger zone for biosignature detection around celestial bodies.
/// Uses a SphereCollider set as trigger to detect when the spacecraft enters.
/// </summary>
public class BiosignatureZone : MonoBehaviour
{
    [Header("Detection Properties")]
    public string biosignatureType;     // "liquid_water", "ice", "organic_compounds", "signs_of_intelligence"
    public float detectionRadius = 2f;
    public CelestialBody parentBody;

    [Header("Visual")]
    public Color zoneColor = new Color(0f, 1f, 0.5f, 0.15f);

    private SphereCollider zoneCollider;
    private bool spacecraftInside = false;

    void Start()
    {
        // Setup trigger collider
        zoneCollider = gameObject.AddComponent<SphereCollider>();
        zoneCollider.isTrigger = true;
        zoneCollider.radius = detectionRadius;

        // Create semi-transparent visual
        CreateVisualIndicator();
    }

    void Update()
    {
        // Follow parent celestial body
        if (parentBody != null)
        {
            transform.position = parentBody.transform.position;
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Spacecraft"))
        {
            spacecraftInside = true;
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Spacecraft"))
        {
            spacecraftInside = false;
        }
    }

    public bool IsSpacecraftInside()
    {
        return spacecraftInside;
    }

    private void CreateVisualIndicator()
    {
        // Create a child sphere for the visual zone indicator
        GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        visual.name = $"Zone_Visual_{biosignatureType}";
        visual.transform.SetParent(transform);
        visual.transform.localPosition = Vector3.zero;
        visual.transform.localScale = Vector3.one * detectionRadius * 2f;

        // Remove the collider from the visual (we use the parent's trigger)
        Destroy(visual.GetComponent<SphereCollider>());

        // Set up transparent material
        Renderer rend = visual.GetComponent<Renderer>();
        if (rend != null)
        {
            Material mat = new Material(Shader.Find("Standard"));
            mat.SetFloat("_Mode", 3); // Transparent mode
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = 3000;
            mat.color = zoneColor;
            rend.material = mat;
        }
    }
}
