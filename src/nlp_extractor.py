import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

# These are the fault signals our system understands.
# Each signal maps to a list of phrases that mean the same thing.
# When any phrase is found in the log text, the signal fires as True.
# This is our "vocabulary" — the bridge between human language and machine state.
SIGNAL_PATTERNS = {
    "high_torque": [
        "high torque",
        "torque detected",
        "mechanical resistance",
        "torque elevated",
    ],
    "low_rpm": [
        "rotational speed dropped",
        "speed dropped",
        "rpm below",
        "low rotational speed",
    ],
    "high_process_temp": [
        "process temperature high",
        "heat buildup",
        "cutting zone",
        "process temp",
    ],
    "high_air_temp": [
        "air temperature elevated",
        "ambient temperature",
        "air temp high",
    ],
    "wear_warning": [
        "tool wear at",
        "approaching replacement",
        "wear approaching",
    ],
    "wear_critical": [
        "tool wear critical",
        "immediate replacement",
        "wear critical",
    ],
    "normal_operation": [
        "normal parameters",
        "no anomalies",
        "operating normally",
    ],
}


def build_matcher():
    # PhraseMatcher lets us search for exact phrases in text efficiently.
    # We build it once and reuse it — important for performance.
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for signal, phrases in SIGNAL_PATTERNS.items():
        patterns = [nlp.make_doc(phrase) for phrase in phrases]
        matcher.add(signal, patterns)
    return matcher


# Build the matcher once when the module loads — not every time we call extract()
MATCHER = build_matcher()


def extract_signals(log_text):
    # Takes a maintenance log string.
    # Returns a dictionary of signals — True if detected, False if not.
    # Example output:
    # {
    #   "high_torque": True,
    #   "low_rpm": False,
    #   "high_process_temp": False,
    #   "high_air_temp": False,
    #   "wear_warning": False,
    #   "wear_critical": True,
    #   "normal_operation": False
    # }

    doc = nlp(log_text.lower())
    matches = MATCHER(doc)

    # Start with all signals as False
    signals = {signal: False for signal in SIGNAL_PATTERNS}

    # Mark each matched signal as True
    for match_id, start, end in matches:
        signal_name = nlp.vocab.strings[match_id]
        signals[signal_name] = True

    return signals


def signals_to_text(signals):
    # Converts the signals dict into a readable summary.
    # Useful for showing the user what the NLP layer detected.
    active = [s for s, v in signals.items() if v and s != "normal_operation"]
    if not active:
        return "No fault signals detected — machine appears healthy."
    readable = {
        "high_torque": "High torque (mechanical resistance)",
        "low_rpm": "Low rotational speed",
        "high_process_temp": "High process temperature (heat buildup)",
        "high_air_temp": "Elevated ambient temperature",
        "wear_warning": "Tool wear approaching limit",
        "wear_critical": "Tool wear critical",
    }
    return " | ".join(readable.get(s, s) for s in active)


if __name__ == "__main__":
    # Test with the exact logs our generator produced
    test_logs = [
        "machine operating within normal parameters. no anomalies detected.",
        "high torque detected at 65.7 Nm indicating mechanical resistance. tool wear critical at 191 minutes — immediate replacement required.",
        "tool wear at 143 minutes approaching replacement window.",
        "process temperature high indicating heat buildup at cutting zone. rotational speed dropped to 1389 rpm below expected threshold.",
        "ambient air temperature elevated beyond normal range. high torque detected at 58.2 Nm indicating mechanical resistance.",
    ]

    print("=== NLP SIGNAL EXTRACTION TEST ===\n")
    for log in test_logs:
        signals = extract_signals(log)
        active = {k: v for k, v in signals.items() if v}
        print(f"Log    : {log}")
        print(f"Signals: {signals_to_text(signals)}")
        print(f"Raw    : {active}")
        print("-" * 70)