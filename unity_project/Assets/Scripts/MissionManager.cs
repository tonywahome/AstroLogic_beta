using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Tracks mission state: biosignatures found/transmitted, terminal conditions.
/// </summary>
public class MissionManager : MonoBehaviour
{
    public static MissionManager Instance { get; private set; }

    [Header("Mission Parameters")]
    public int requiredBiosignatures = 3;
    public float maxDistance = 50f;          // AU (in world units / distanceScale)
    public int maxSteps = 100000;

    // Mission state
    public HashSet<string> BiosignaturesFound { get; private set; } = new HashSet<string>();
    public HashSet<string> BiosignaturesTransmitted { get; private set; } = new HashSet<string>();
    public int CurrentStep { get; private set; }
    public float CumulativeReward { get; private set; }

    // Terminal state
    public bool IsSuccess { get; private set; }
    public bool IsCollision { get; private set; }
    public bool IsOutOfBounds { get; private set; }
    public bool IsResourceDepleted { get; private set; }
    public bool IsMaxSteps { get; private set; }

    void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    public void ResetMission()
    {
        BiosignaturesFound = new HashSet<string>();
        BiosignaturesTransmitted = new HashSet<string>();
        CurrentStep = 0;
        CumulativeReward = 0f;
        IsSuccess = false;
        IsCollision = false;
        IsOutOfBounds = false;
        IsResourceDepleted = false;
        IsMaxSteps = false;
    }

    public void IncrementStep()
    {
        CurrentStep++;
        if (CurrentStep >= maxSteps)
            IsMaxSteps = true;
    }

    public void AddReward(float reward)
    {
        CumulativeReward += reward;
    }

    /// <summary>
    /// Record a newly detected biosignature.
    /// Returns true if this is a new detection.
    /// </summary>
    public bool RecordBiosignature(string type)
    {
        return BiosignaturesFound.Add(type);
    }

    /// <summary>
    /// Transmit all found biosignatures.
    /// Returns list of newly transmitted ones.
    /// </summary>
    public List<string> TransmitBiosignatures()
    {
        List<string> newTransmissions = new List<string>();
        foreach (string biosig in BiosignaturesFound)
        {
            if (BiosignaturesTransmitted.Add(biosig))
            {
                newTransmissions.Add(biosig);
            }
        }

        if (BiosignaturesTransmitted.Count >= requiredBiosignatures)
            IsSuccess = true;

        return newTransmissions;
    }

    public void SetCollision() { IsCollision = true; }
    public void SetOutOfBounds() { IsOutOfBounds = true; }
    public void SetResourceDepleted() { IsResourceDepleted = true; }

    public bool IsTerminated()
    {
        return IsSuccess || IsCollision || IsOutOfBounds || IsResourceDepleted || IsMaxSteps;
    }
}
