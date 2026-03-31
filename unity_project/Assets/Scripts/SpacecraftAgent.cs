using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;

/// <summary>
/// ML-Agents spacecraft agent for astrobiological exploration.
/// Navigates the solar system to detect and transmit biosignatures.
///
/// Observation Space (23-dim):
///   [0:3]   Position (x, y, z) in AU
///   [3:6]   Velocity (vx, vy, vz)
///   [6]     Normalized distance to Mars
///   [7:10]  Heading to Mars (unit vector)
///   [10]    Normalized distance to Europa
///   [11:14] Heading to Europa
///   [14]    Normalized distance to Enceladus
///   [15:18] Heading to Enceladus
///   [18]    Fuel [0,1]
///   [19]    Battery [0,1]
///   [20]    SNR biosignature signal [0,1]
///   [21]    Biosignatures found / 3
///   [22]    Biosignatures transmitted / 3
///
/// Action Space (MultiDiscrete [5, 3, 3, 3, 4, 2]):
///   [0] Thrust:     {0, 0.25, 0.5, 0.75, 1.0}
///   [1] Pitch:      {-5, 0, +5} degrees
///   [2] Yaw:        {-5, 0, +5} degrees
///   [3] Roll:       {-5, 0, +5} degrees
///   [4] Instrument: {None, Spectrometer, ThermalImager, Drill}
///   [5] Comm:       {Off, Transmit}
/// </summary>
public class SpacecraftAgent : Agent
{
    [Header("Mission Parameters")]
    public float maxThrust = 0.001f;
    public float rotationSpeed = 5f;          // degrees per step
    public float fuelConsumptionRate = 0.0005f;
    public float batteryDrainRate = 0.00005f;
    public float solarRechargeRange = 1.5f;   // AU
    public float solarRechargeRate = 0.0001f;
    public float snrDetectionThreshold = 0.5f;
    public float maxDistance = 50f;            // AU boundary

    [Header("Resources")]
    public float fuel = 1.0f;
    public float battery = 1.0f;

    [Header("References")]
    public SolarSystemManager solarSystem;
    public MissionManager missionManager;

    // State
    private Rigidbody rb;
    private Vector3 orientation;              // pitch, yaw, roll in radians
    private int activeInstrument = 0;
    private bool isTransmitting = false;
    private float distanceScale;

    // Thrust levels
    private readonly float[] thrustLevels = { 0f, 0.25f, 0.5f, 0.75f, 1.0f };
    private readonly float[] rotationDeltas = { -5f, 0f, 5f };

    // Instrument definitions: which biosignatures each can detect
    private readonly Dictionary<int, string[]> instrumentDetects = new Dictionary<int, string[]>
    {
        { 0, new string[] { } },                                                    // None
        { 1, new string[] { "liquid_water", "organic_compounds" } },                // Spectrometer
        { 2, new string[] { "ice", "liquid_water" } },                              // ThermalImager
        { 3, new string[] { "organic_compounds", "ice", "signs_of_intelligence" } } // Drill
    };

    private readonly string[] instrumentNames = { "None", "Spectrometer", "ThermalImager", "Drill" };

    // Target body names matching Python env ordering
    private readonly string[] targetNames = { "Mars", "Europa", "Enceladus" };

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        rb.useGravity = false;
        rb.linearDamping = 0f;
        rb.angularDamping = 0f;

        if (solarSystem == null)
            solarSystem = FindObjectOfType<SolarSystemManager>();
        if (missionManager == null)
            missionManager = FindObjectOfType<MissionManager>();

        distanceScale = solarSystem != null ? solarSystem.distanceScale : 10f;
    }

    public override void OnEpisodeBegin()
    {
        // Reset to Earth's starting position
        CelestialBody earth = null;
        if (solarSystem != null && solarSystem.Bodies.ContainsKey("Earth"))
            earth = solarSystem.Bodies["Earth"];

        Vector3 startPos = earth != null ? earth.transform.position : new Vector3(distanceScale, 0, 0);

        transform.position = startPos;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // Earth's orbital velocity (tangential)
        float v_earth = 2f * Mathf.PI * 1.0f / 365.25f; // AU/day
        rb.linearVelocity = new Vector3(0, 0, v_earth * distanceScale);

        orientation = Vector3.zero;
        fuel = 1.0f;
        battery = 1.0f;
        activeInstrument = 0;
        isTransmitting = false;

        if (missionManager != null)
            missionManager.ResetMission();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Position (in AU, convert from world units)
        Vector3 posAU = transform.position / distanceScale;
        sensor.AddObservation(posAU);                          // [0:3]

        // Velocity
        Vector3 velAU = rb.linearVelocity / distanceScale;
        sensor.AddObservation(Vector3.ClampMagnitude(velAU, 10f)); // [3:6]

        // Target body data (Mars, Europa, Enceladus)
        foreach (string targetName in targetNames)
        {
            if (solarSystem != null && solarSystem.Bodies.ContainsKey(targetName))
            {
                CelestialBody target = solarSystem.Bodies[targetName];
                Vector3 toTarget = target.transform.position - transform.position;
                float distAU = toTarget.magnitude / distanceScale;

                // Normalized distance (closer = higher)
                float normDist = Mathf.Max(0f, 1f - distAU / maxDistance);
                sensor.AddObservation(normDist);                // [6/10/14]

                // Heading unit vector
                Vector3 heading = toTarget.normalized;
                sensor.AddObservation(heading);                 // [7:10/11:14/15:18]
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(Vector3.zero);
            }
        }

        // Resources
        sensor.AddObservation(fuel);                            // [18]
        sensor.AddObservation(battery);                         // [19]

        // SNR
        float snr = solarSystem != null ? solarSystem.GetMaxSNR(transform.position) : 0f;
        sensor.AddObservation(snr);                             // [20]

        // Mission progress
        int found = missionManager != null ? missionManager.BiosignaturesFound.Count : 0;
        int transmitted = missionManager != null ? missionManager.BiosignaturesTransmitted.Count : 0;
        sensor.AddObservation(found / 3f);                      // [21]
        sensor.AddObservation(transmitted / 3f);                // [22]
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int thrustIdx = actions.DiscreteActions[0];
        int pitchIdx = actions.DiscreteActions[1];
        int yawIdx = actions.DiscreteActions[2];
        int rollIdx = actions.DiscreteActions[3];
        int instrumentIdx = actions.DiscreteActions[4];
        int commIdx = actions.DiscreteActions[5];

        float thrustLevel = thrustLevels[thrustIdx];
        float pitchDelta = rotationDeltas[pitchIdx] * Mathf.Deg2Rad;
        float yawDelta = rotationDeltas[yawIdx] * Mathf.Deg2Rad;
        float rollDelta = rotationDeltas[rollIdx] * Mathf.Deg2Rad;
        activeInstrument = instrumentIdx;
        isTransmitting = commIdx == 1;

        // Update orientation
        orientation.x += pitchDelta;
        orientation.y += yawDelta;
        orientation.z += rollDelta;

        // Apply thrust
        float fuelUsed = 0f;
        if (thrustLevel > 0 && fuel > 0)
        {
            Vector3 forward = OrientationToDirection(orientation);
            Vector3 thrustAccel = forward * thrustLevel * maxThrust * distanceScale;
            rb.AddForce(thrustAccel, ForceMode.Acceleration);
            fuelUsed = thrustLevel * fuelConsumptionRate;
            fuel = Mathf.Max(0f, fuel - fuelUsed);
        }

        // Apply gravity
        if (solarSystem != null)
        {
            Vector3 gravAccel = solarSystem.GetTotalGravity(transform.position, rb.mass);
            rb.AddForce(gravAccel * distanceScale, ForceMode.Acceleration);
        }

        // Battery management
        battery -= batteryDrainRate;
        float sunDist = transform.position.magnitude / distanceScale;
        if (sunDist < solarRechargeRange)
            battery += solarRechargeRate;
        battery = Mathf.Clamp01(battery);

        // Update orbits
        if (solarSystem != null)
            solarSystem.UpdateOrbits();

        // Compute SNR
        float snr = solarSystem != null ? solarSystem.GetMaxSNR(transform.position) : 0f;

        // Biosignature detection
        List<string> newBiosignatures = new List<string>();
        if (instrumentIdx > 0 && snr >= snrDetectionThreshold)
        {
            newBiosignatures = AttemptDetection(instrumentIdx);
        }

        // Communication
        List<string> newTransmissions = new List<string>();
        if (isTransmitting && missionManager != null)
        {
            newTransmissions = missionManager.TransmitBiosignatures();
        }

        // Rewards
        float reward = 0f;

        // Step penalty (dense)
        reward -= 0.01f * fuelUsed + 0.001f;

        // Biosignature detection rewards
        foreach (string biosig in newBiosignatures)
        {
            switch (biosig)
            {
                case "liquid_water": reward += 500f; break;
                case "ice": reward += 300f; break;
                case "organic_compounds": reward += 750f; break;
                case "signs_of_intelligence": reward += 5000f; break;
            }
        }

        // Transmission bonus
        reward += newTransmissions.Count * 50f;

        // Proximity shaping
        if (solarSystem != null)
        {
            float minDist = solarSystem.GetMinTargetDistance(transform.position) / distanceScale;
            if (minDist < 5f)
            {
                float proximityReward = 0.1f * (1f / (minDist + 0.1f) - 1f / 5.1f);
                reward += Mathf.Max(0f, proximityReward);
            }
        }

        // Check terminal conditions
        bool collision = solarSystem != null && solarSystem.CheckCollision(transform.position);
        bool outOfBounds = transform.position.magnitude / distanceScale > maxDistance;
        bool resourceDepleted = fuel <= 0 || battery <= 0;

        if (collision && missionManager != null) missionManager.SetCollision();
        if (outOfBounds && missionManager != null) missionManager.SetOutOfBounds();
        if (resourceDepleted && missionManager != null) missionManager.SetResourceDepleted();

        if (collision || outOfBounds)
            reward += -1000f;

        AddReward(reward);
        if (missionManager != null)
        {
            missionManager.IncrementStep();
            missionManager.AddReward(reward);
        }

        // Update UI
        if (UIManager.Instance != null)
        {
            int found = missionManager != null ? missionManager.BiosignaturesFound.Count : 0;
            int transmitted = missionManager != null ? missionManager.BiosignaturesTransmitted.Count : 0;
            int step = missionManager != null ? missionManager.CurrentStep : 0;
            float cumReward = missionManager != null ? missionManager.CumulativeReward : 0f;

            UIManager.Instance.UpdateHUD(
                fuel, battery,
                instrumentNames[activeInstrument], isTransmitting,
                found, transmitted,
                step, 100000,
                cumReward,
                transform.position / distanceScale,
                rb.linearVelocity.magnitude / distanceScale,
                snr
            );
        }

        // End episode if terminated
        if (missionManager != null && missionManager.IsTerminated())
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var d = actionsOut.DiscreteActions;

        // Thrust: W = forward, S = none
        d[0] = Input.GetKey(KeyCode.W) ? 4 : 0;
        // Pitch: Up/Down arrows
        d[1] = Input.GetKey(KeyCode.UpArrow) ? 2 : (Input.GetKey(KeyCode.DownArrow) ? 0 : 1);
        // Yaw: A/D
        d[2] = Input.GetKey(KeyCode.D) ? 2 : (Input.GetKey(KeyCode.A) ? 0 : 1);
        // Roll: Q/E
        d[3] = Input.GetKey(KeyCode.E) ? 2 : (Input.GetKey(KeyCode.Q) ? 0 : 1);
        // Instrument: 1-4
        d[4] = Input.GetKey(KeyCode.Alpha1) ? 1 :
               Input.GetKey(KeyCode.Alpha2) ? 2 :
               Input.GetKey(KeyCode.Alpha3) ? 3 : 0;
        // Communication: Space
        d[5] = Input.GetKey(KeyCode.Space) ? 1 : 0;
    }

    private List<string> AttemptDetection(int instrumentIdx)
    {
        List<string> detected = new List<string>();
        if (solarSystem == null || missionManager == null) return detected;

        string[] canDetect = instrumentDetects[instrumentIdx];

        foreach (var target in solarSystem.TargetBodies)
        {
            if (!target.IsInDetectionZone(transform.position)) continue;

            foreach (string biosig in target.biosignatures)
            {
                bool instrumentCanDetect = System.Array.Exists(canDetect, s => s == biosig);
                if (instrumentCanDetect && missionManager.RecordBiosignature(biosig))
                {
                    detected.Add(biosig);
                }
            }
        }
        return detected;
    }

    private Vector3 OrientationToDirection(Vector3 orient)
    {
        float cosYaw = Mathf.Cos(orient.y);
        float sinYaw = Mathf.Sin(orient.y);
        float cosPitch = Mathf.Cos(orient.x);
        float sinPitch = Mathf.Sin(orient.x);

        return new Vector3(
            cosPitch * cosYaw,
            sinPitch,
            cosPitch * sinYaw
        ).normalized;
    }
}
