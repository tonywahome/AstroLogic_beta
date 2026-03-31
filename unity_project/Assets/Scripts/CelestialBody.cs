using UnityEngine;

/// <summary>
/// Represents a celestial body with orbital motion and gravitational influence.
/// Supports both bodies orbiting the Sun and moons orbiting parent planets.
/// </summary>
public class CelestialBody : MonoBehaviour
{
    [Header("Physical Properties")]
    public float mass = 1.0f;               // Relative mass (solar masses)
    public float bodyRadius = 0.5f;         // Visual/collision radius in world units

    [Header("Orbital Parameters")]
    public float orbitRadius = 10f;         // Distance from parent body
    public float orbitPeriod = 365.25f;     // Orbital period in simulation days
    public float initialAngle = 0f;         // Starting orbital angle (radians)

    [Header("References")]
    public Transform parentBody;            // What this body orbits (null for Sun)

    [Header("Biosignatures")]
    public string[] biosignatures;          // Types of biosignatures present
    public float detectionZoneRadius = 2f;  // Detection zone radius in world units

    [Header("Visual")]
    public Color bodyColor = Color.white;

    private float currentAngle;
    private float simulationTime;

    void Start()
    {
        currentAngle = initialAngle;
        simulationTime = 0f;

        // Set material color
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material.color = bodyColor;
            // Make the Sun emissive
            if (parentBody == null && orbitRadius == 0f)
            {
                rend.material.EnableKeyword("_EMISSION");
                rend.material.SetColor("_EmissionColor", bodyColor * 2f);
            }
        }
    }

    /// <summary>
    /// Update the body's orbital position based on elapsed simulation time.
    /// </summary>
    public void UpdateOrbitalPosition(float deltaTime)
    {
        if (orbitPeriod <= 0 || orbitRadius <= 0) return;

        simulationTime += deltaTime;
        currentAngle = initialAngle + 2f * Mathf.PI * simulationTime / orbitPeriod;

        Vector3 orbitalPos = new Vector3(
            orbitRadius * Mathf.Cos(currentAngle),
            0f,
            orbitRadius * Mathf.Sin(currentAngle)
        );

        if (parentBody != null)
            transform.position = parentBody.position + orbitalPos;
        else
            transform.position = orbitalPos;
    }

    /// <summary>
    /// Compute gravitational force exerted on a spacecraft at the given position.
    /// F = G * M * m / r^2, directed toward this body.
    /// </summary>
    public Vector3 GetGravitationalForce(Vector3 spacecraftPos, float spacecraftMass)
    {
        Vector3 direction = transform.position - spacecraftPos;
        float distance = direction.magnitude;

        if (distance < 0.01f) return Vector3.zero;

        // G normalized: using 4*pi^2 in simulation units
        float G = 4f * Mathf.PI * Mathf.PI / (365.25f * 365.25f);
        float forceMagnitude = G * mass * spacecraftMass / (distance * distance);

        return direction.normalized * forceMagnitude;
    }

    /// <summary>
    /// Check if a position is within the collision radius of this body.
    /// </summary>
    public bool IsColliding(Vector3 position)
    {
        float dist = Vector3.Distance(transform.position, position);
        return dist < bodyRadius;
    }

    /// <summary>
    /// Check if a position is within the biosignature detection zone.
    /// </summary>
    public bool IsInDetectionZone(Vector3 position)
    {
        float dist = Vector3.Distance(transform.position, position);
        return dist < detectionZoneRadius;
    }

    /// <summary>
    /// Get the signal-to-noise ratio for biosignature detection at a position.
    /// Returns 0-1, where 1 is at the center and 0 is at the zone boundary.
    /// </summary>
    public float GetSNR(Vector3 position)
    {
        if (detectionZoneRadius <= 0) return 0f;
        float dist = Vector3.Distance(transform.position, position);
        if (dist >= detectionZoneRadius) return 0f;
        return 1f - (dist / detectionZoneRadius);
    }
}
