using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// HUD overlay displaying mission telemetry: fuel, battery, biosignature status,
/// instrument info, step counter, and cumulative reward.
/// </summary>
public class UIManager : MonoBehaviour
{
    public static UIManager Instance { get; private set; }

    [Header("UI References")]
    public Slider fuelBar;
    public Slider batteryBar;
    public Text fuelText;
    public Text batteryText;
    public Text instrumentText;
    public Text commStatusText;
    public Text biosigText;
    public Text stepCounterText;
    public Text rewardText;
    public Text positionText;
    public Text velocityText;
    public Text snrText;

    void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    public void UpdateHUD(
        float fuel, float battery,
        string activeInstrument, bool isTransmitting,
        int biosigFound, int biosigTransmitted,
        int currentStep, int maxSteps,
        float cumulativeReward,
        Vector3 position, float velocityMagnitude, float snr)
    {
        if (fuelBar != null)
        {
            fuelBar.value = fuel;
            // Color gradient: green -> yellow -> red
            var fuelFill = fuelBar.fillRect?.GetComponent<Image>();
            if (fuelFill != null)
                fuelFill.color = Color.Lerp(Color.red, Color.green, fuel);
        }

        if (batteryBar != null)
            batteryBar.value = battery;

        if (fuelText != null)
            fuelText.text = $"FUEL: {fuel:P1}";

        if (batteryText != null)
            batteryText.text = $"BATT: {battery:P1}";

        if (instrumentText != null)
            instrumentText.text = $"INSTRUMENT: {activeInstrument}";

        if (commStatusText != null)
        {
            commStatusText.text = isTransmitting ? "TRANSMITTING" : "IDLE";
            commStatusText.color = isTransmitting ? Color.green : Color.gray;
        }

        if (biosigText != null)
            biosigText.text = $"FOUND: {biosigFound}  TX: {biosigTransmitted}";

        if (stepCounterText != null)
            stepCounterText.text = $"Step: {currentStep:N0} / {maxSteps:N0}";

        if (rewardText != null)
            rewardText.text = $"Reward: {cumulativeReward:F2}";

        if (positionText != null)
            positionText.text = $"POS: ({position.x:F2}, {position.y:F2}, {position.z:F2})";

        if (velocityText != null)
            velocityText.text = $"VEL: {velocityMagnitude:F4} AU/day";

        if (snrText != null)
        {
            snrText.text = $"SNR: {snr:F3}";
            snrText.color = snr > 0.5f ? Color.green : (snr > 0.1f ? Color.yellow : Color.gray);
        }
    }
}
