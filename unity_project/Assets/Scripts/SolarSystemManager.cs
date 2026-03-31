using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Manages the solar system: spawns celestial bodies, updates orbits,
/// and provides access to body references for the agent.
/// </summary>
public class SolarSystemManager : MonoBehaviour
{
    public static SolarSystemManager Instance { get; private set; }

    [Header("Simulation")]
    public float timeScale = 1f;            // Simulation speed multiplier
    public float simulationDeltaTime = 1f;  // Time step in days

    [Header("Scale")]
    public float distanceScale = 10f;       // World units per AU
    public float bodyScaleMultiplier = 5f;  // Exaggeration for visibility

    // References to all celestial bodies
    public Dictionary<string, CelestialBody> Bodies { get; private set; } = new Dictionary<string, CelestialBody>();
    public List<CelestialBody> TargetBodies { get; private set; } = new List<CelestialBody>();

    void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    /// <summary>
    /// Register a celestial body with the manager.
    /// </summary>
    public void RegisterBody(string name, CelestialBody body)
    {
        Bodies[name] = body;
        if (body.biosignatures != null && body.biosignatures.Length > 0)
        {
            TargetBodies.Add(body);
        }
    }

    /// <summary>
    /// Update all orbital positions.
    /// </summary>
    public void UpdateOrbits()
    {
        float dt = simulationDeltaTime * timeScale;

        // Update planets first (orbit Sun)
        foreach (var kvp in Bodies)
        {
            CelestialBody body = kvp.Value;
            if (body.parentBody == null || kvp.Key == "Sun") continue;
            // Skip moons on first pass
            string parentName = GetParentName(kvp.Key);
            if (parentName == "Sun")
            {
                body.UpdateOrbitalPosition(dt);
            }
        }

        // Then update moons (orbit planets)
        foreach (var kvp in Bodies)
        {
            CelestialBody body = kvp.Value;
            string parentName = GetParentName(kvp.Key);
            if (parentName != null && parentName != "Sun")
            {
                body.UpdateOrbitalPosition(dt);
            }
        }
    }

    /// <summary>
    /// Get total gravitational acceleration at a position from all bodies.
    /// </summary>
    public Vector3 GetTotalGravity(Vector3 position, float spacecraftMass)
    {
        Vector3 totalForce = Vector3.zero;
        foreach (var kvp in Bodies)
        {
            totalForce += kvp.Value.GetGravitationalForce(position, spacecraftMass);
        }
        return totalForce / spacecraftMass; // Return acceleration
    }

    /// <summary>
    /// Check if position collides with any celestial body.
    /// </summary>
    public bool CheckCollision(Vector3 position)
    {
        foreach (var kvp in Bodies)
        {
            if (kvp.Value.IsColliding(position))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Get the maximum SNR at a position from any target body.
    /// </summary>
    public float GetMaxSNR(Vector3 position)
    {
        float maxSNR = 0f;
        foreach (var target in TargetBodies)
        {
            float snr = target.GetSNR(position);
            maxSNR = Mathf.Max(maxSNR, snr);
        }
        return maxSNR;
    }

    /// <summary>
    /// Get minimum distance to any target body.
    /// </summary>
    public float GetMinTargetDistance(Vector3 position)
    {
        float minDist = float.MaxValue;
        foreach (var target in TargetBodies)
        {
            float dist = Vector3.Distance(position, target.transform.position);
            minDist = Mathf.Min(minDist, dist);
        }
        return minDist;
    }

    private string GetParentName(string bodyName)
    {
        switch (bodyName)
        {
            case "Europa": return "Jupiter";
            case "Enceladus": return "Saturn";
            case "Earth":
            case "Mars":
            case "Jupiter":
            case "Saturn":
                return "Sun";
            default: return null;
        }
    }
}
