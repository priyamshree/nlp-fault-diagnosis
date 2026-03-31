import pandas as pd

THRESHOLDS = {
    "air_temp_high": 302.0,
    "process_temp_high": 311.0,
    "rpm_low": 1400,
    "torque_high": 55.0,
    "wear_warning": 100,
    "wear_critical": 180,
}

FAILURE_LABELS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

# These phrases are added when a specific failure type is confirmed.
# This is what makes each failure class linguistically distinct —
# a real operator observing a power failure would mention electrical
# symptoms, not just sensor readings.
FAILURE_PHRASES = {
    "TWF": [
        "tool surface showing progressive edge degradation",
        "cutting resistance increasing with each pass",
        "tool tip visibly worn under inspection",
    ],
    "HDF": [
        "coolant flow reduced — check pump and filter",
        "chip evacuation insufficient causing thermal buildup",
        "cooling fins blocked — ambient heat rising steadily",
    ],
    "PWF": [
        "motor drawing irregular current at startup",
        "voltage fluctuation detected on spindle drive",
        "power consumption spiking beyond rated load",
    ],
    "OSF": [
        "spindle load exceeding rated mechanical limits",
        "cutting assembly vibrating under excessive feed force",
        "bearing shows stress marks from repeated overload",
    ],
    "RNF": [
        "intermittent behaviour with no repeatable pattern",
        "sporadic reading inconsistent with sensor history",
        "unexplained variation in output across cycles",
    ],
}


def get_failure_type(row):
    for col, label in FAILURE_LABELS.items():
        if row[col] == 1:
            return label
    return "No Failure"


def get_failure_key(row):
    for col in FAILURE_LABELS:
        if row[col] == 1:
            return col
    return None


def generate_log(row):
    observations = []

    # Sensor-based observations — same as before
    if row["Air temperature [K]"] > THRESHOLDS["air_temp_high"]:
        observations.append(
            "ambient air temperature elevated beyond normal range"
        )

    if row["Process temperature [K]"] > THRESHOLDS["process_temp_high"]:
        observations.append(
            "process temperature high indicating heat buildup at cutting zone"
        )

    if row["Rotational speed [rpm]"] < THRESHOLDS["rpm_low"]:
        observations.append(
            f"rotational speed dropped to {int(row['Rotational speed [rpm]'])} rpm "
            f"below expected threshold"
        )

    if row["Torque [Nm]"] > THRESHOLDS["torque_high"]:
        observations.append(
            f"high torque detected at {row['Torque [Nm]']:.1f} Nm "
            f"indicating mechanical resistance"
        )

    wear = row["Tool wear [min]"]
    if wear > THRESHOLDS["wear_critical"]:
        observations.append(
            f"tool wear critical at {int(wear)} minutes — "
            f"immediate replacement required"
        )
    elif wear > THRESHOLDS["wear_warning"]:
        observations.append(
            f"tool wear at {int(wear)} minutes approaching replacement window"
        )

    # Failure-type-specific language — this is what teaches the classifier
    # to distinguish between failure types that have similar sensor readings.
    failure_key = get_failure_key(row)
    if failure_key and failure_key in FAILURE_PHRASES:
        # Add the first distinctive phrase for this failure type
        observations.append(FAILURE_PHRASES[failure_key][0])

    if not observations:
        return "machine operating within normal parameters. no anomalies detected."

    return ". ".join(observations) + "."


def generate_all_logs(csv_path):
    df = pd.read_csv(csv_path)
    df["failure_type"] = df.apply(get_failure_type, axis=1)
    df["maintenance_log"] = df.apply(generate_log, axis=1)
    return df


if __name__ == "__main__":
    df = generate_all_logs("data/ai4i2020.csv")

    print("=== FAILURE TYPE COUNTS ===")
    print(df["failure_type"].value_counts())

    print("\n=== SAMPLE LOGS PER FAILURE TYPE ===\n")
    for fault_key, fault_label in FAILURE_LABELS.items():
        rows = df[df["failure_type"] == fault_label]
        if len(rows) > 0:
            row = rows.iloc[0]
            print(f"Type : {fault_label}")
            print(f"Log  : {row['maintenance_log']}")
            print("-" * 70)

    normal = df[df["failure_type"] == "No Failure"].iloc[0]
    print(f"Type : No Failure")
    print(f"Log  : {normal['maintenance_log']}")