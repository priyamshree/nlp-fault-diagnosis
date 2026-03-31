import gradio as gr
import pandas as pd
from src.text_generator import generate_all_logs
from src.nlp_extractor import extract_signals, signals_to_text
from src.fault_graph import build_graph, get_active_signals, astar_diagnosis

G = build_graph()
df = generate_all_logs("data/ai4i2020.csv")

SAMPLES = {}
for fault in ["Tool Wear Failure", "Heat Dissipation Failure",
              "Power Failure", "Overstrain Failure", "No Failure"]:
    row = df[df["failure_type"] == fault].iloc[0]
    SAMPLES[fault] = row["maintenance_log"]

RECOMMENDATIONS = {
    "Tool Wear Failure":
        "🔧 Replace cutting tool immediately. Inspect spindle for secondary damage.",
    "Heat Dissipation Failure":
        "🌡️ Check coolant flow and fan systems. Reduce process temperature before resuming.",
    "Power Failure":
        "⚡ Inspect power supply and motor windings. Check for voltage irregularities.",
    "Overstrain Failure":
        "⚙️ Reduce feed rate and cutting depth. Inspect spindle bearing for damage.",
    "Random Failure":
        "🔍 No clear pattern identified. Run full diagnostic inspection of all subsystems.",
    "No Failure":
        "✅ Machine operating normally. Continue standard monitoring schedule.",
}

FAULT_COLORS = {
    "Tool Wear Failure": "🟠",
    "Heat Dissipation Failure": "🔴",
    "Power Failure": "🟡",
    "Overstrain Failure": "🔴",
    "Random Failure": "⚪",
    "No Failure": "🟢",
}

CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
.output-box textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
}
.diagnosis-label {
    font-size: 18px !important;
    font-weight: 600 !important;
}
#run-btn {
    background: #1a2744 !important;
    color: white !important;
    border: none !important;
    font-size: 15px !important;
    padding: 12px !important;
    border-radius: 8px !important;
}
#run-btn:hover {
    background: #253460 !important;
}
.fault-card {
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
footer { display: none !important; }
"""

def diagnose(log_text):
    if not log_text.strip():
        return (
            "—", "—", "Please enter a maintenance log to begin.",
            "", build_empty_report()
        )

    signals = extract_signals(log_text)
    signals_summary = signals_to_text(signals)
    active = get_active_signals(signals)
    fault, confidence, evidence = astar_diagnosis(G, active)

    icon = FAULT_COLORS.get(fault, "⚪")
    fault_display = f"{icon}  {fault}"
    confidence_display = f"{confidence}%"

    if evidence:
        evidence_lines = "\n".join(
            f"  {'█' * int(w * 10)}{'░' * (10 - int(w * 10))}  "
            f"{sig.replace('_', ' ').title():<28} {round(w * 100)}%"
            for sig, w in evidence
        )
    else:
        evidence_lines = "  No specific fault signals detected."

    active_signals_display = signals_summary if signals_summary else "None"

    report = f"""╔══════════════════════════════════════════════╗
║           DIAGNOSIS REPORT                   ║
╚══════════════════════════════════════════════╝

  Root Cause   :  {fault}
  Confidence   :  {confidence}%
  Signals Found:  {len(active)} active

──────────────────────────────────────────────
  EVIDENCE TRAIL
──────────────────────────────────────────────
{evidence_lines}

──────────────────────────────────────────────
  RECOMMENDATION
──────────────────────────────────────────────
  {RECOMMENDATIONS.get(fault, 'Consult maintenance manual.')}

──────────────────────────────────────────────
  SEARCH PATH (A* Traversal)
──────────────────────────────────────────────
  Signals → [{' , '.join(active)}]
       ↓
  Fault Graph (13 nodes, 16 edges)
       ↓
  Best Path → {fault} (cost: {round(1 - confidence/100, 3)})
"""

    return (fault_display, confidence_display,
            active_signals_display, active_signals_display, report)


def build_empty_report():
    return """╔══════════════════════════════════════════════╗
║           DIAGNOSIS REPORT                   ║
╚══════════════════════════════════════════════╝

  Awaiting maintenance log input...

  ➜ Select a sample from the dropdown above, or
    type a custom maintenance log and click
    "Run Diagnosis" to begin.
"""


def load_sample(fault_type):
    return SAMPLES.get(fault_type, "")


with gr.Blocks(title="NLP Fault Diagnosis", css=CSS) as app:

    gr.HTML("""
    <div style="text-align:center; padding: 28px 0 8px 0;">
        <div style="font-size:11px; letter-spacing:0.15em; text-transform:uppercase;
                    color:#9298a8; font-family:monospace; margin-bottom:10px;">
            21CSE356T · Natural Language Processing · SRM IST
        </div>
        <h1 style="font-size:28px; font-weight:700; color:#1a2744;
                   letter-spacing:-0.02em; margin:0 0 8px 0;">
            NLP Fault Diagnosis System
        </h1>
        <p style="color:#5a5f70; font-size:14px; max-width:600px;
                  margin:0 auto; line-height:1.6;">
            Paste an industrial maintenance log. The pipeline extracts fault signals
            using NLP, builds a state graph, and runs A* search to diagnose the root cause.
        </p>
    </div>
    """)

    gr.HTML("""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
                gap:10px; max-width:700px; margin:0 auto 24px auto;">
        <div style="background:#f0f3fa; border-radius:8px; padding:12px;
                    text-align:center; border:1px solid #c8d4ec;">
            <div style="font-size:20px; font-weight:700; color:#1a2744;">5</div>
            <div style="font-size:10px; color:#6a7090; text-transform:uppercase;
                        letter-spacing:0.08em; font-family:monospace;">Pipeline Stages</div>
        </div>
        <div style="background:#f0f3fa; border-radius:8px; padding:12px;
                    text-align:center; border:1px solid #c8d4ec;">
            <div style="font-size:20px; font-weight:700; color:#1a2744;">13</div>
            <div style="font-size:10px; color:#6a7090; text-transform:uppercase;
                        letter-spacing:0.08em; font-family:monospace;">Graph Nodes</div>
        </div>
        <div style="background:#f0f3fa; border-radius:8px; padding:12px;
                    text-align:center; border:1px solid #c8d4ec;">
            <div style="font-size:20px; font-weight:700; color:#1a2744;">10K</div>
            <div style="font-size:10px; color:#6a7090; text-transform:uppercase;
                        letter-spacing:0.08em; font-family:monospace;">Dataset Rows</div>
        </div>
        <div style="background:#f0f3fa; border-radius:8px; padding:12px;
                    text-align:center; border:1px solid #c8d4ec;">
            <div style="font-size:20px; font-weight:700; color:#1a2744;">A*</div>
            <div style="font-size:10px; color:#6a7090; text-transform:uppercase;
                        letter-spacing:0.08em; font-family:monospace;">Search Algorithm</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div style="font-size:13px; font-weight:600; color:#1a2744; '
                    'margin-bottom:8px; text-transform:uppercase; '
                    'letter-spacing:0.08em; font-family:monospace;">① Input</div>')

            sample_picker = gr.Dropdown(
                choices=list(SAMPLES.keys()),
                label="Load sample from dataset",
                value=None,
                info="Pick a pre-loaded example to see how the system works"
            )

            log_input = gr.Textbox(
                label="Maintenance Log",
                placeholder=(
                    "e.g. high torque detected at 65.7 Nm indicating mechanical resistance. "
                    "tool wear critical at 191 minutes — immediate replacement required."
                ),
                lines=6,
                info="Type or paste any industrial maintenance log text"
            )

            submit_btn = gr.Button(
                "⚙  Run Diagnosis",
                variant="primary",
                elem_id="run-btn",
                size="lg"
            )

            gr.HTML("""
            <div style="margin-top:16px; background:#f8f9fc; border-radius:8px;
                        padding:14px 16px; border:1px solid #e0e4ee;">
                <div style="font-size:11px; font-weight:600; color:#5a5f70;
                            text-transform:uppercase; letter-spacing:0.08em;
                            margin-bottom:8px; font-family:monospace;">How it works</div>
                <div style="font-size:12px; color:#5a5f70; line-height:1.8;">
                    ① NLP extracts fault signals from your text<br/>
                    ② Signals mapped onto a fault state graph<br/>
                    ③ A* search finds the lowest-cost fault path<br/>
                    ④ Diagnosis + evidence trail returned
                </div>
            </div>
            """)

        with gr.Column(scale=1):
            gr.HTML('<div style="font-size:13px; font-weight:600; color:#1a2744; '
                    'margin-bottom:8px; text-transform:uppercase; '
                    'letter-spacing:0.08em; font-family:monospace;">② Diagnosis</div>')

            with gr.Row():
                fault_output = gr.Textbox(
                    label="Root Cause",
                    interactive=False,
                    elem_classes=["diagnosis-label"]
                )
                confidence_output = gr.Textbox(
                    label="Confidence",
                    interactive=False,
                    scale=0,
                    min_width=100
                )

            signals_output = gr.Textbox(
                label="Detected Fault Signals",
                interactive=False,
                lines=2
            )

            report_output = gr.Textbox(
                label="Full Diagnosis Report",
                lines=16,
                interactive=False,
                elem_classes=["output-box"]
            )

    gr.HTML("""
    <div style="margin-top:8px; padding:16px 0; border-top:1px solid #e8e5de;
                display:grid; grid-template-columns:1fr 1fr 1fr;
                font-size:11px; color:#9298a8; font-family:monospace;">
        <div>Dataset: AI4I 2020 · UCI ML Repository</div>
        <div style="text-align:center;">
            spaCy · NetworkX · Gradio · Python
        </div>
        <div style="text-align:right;">
            Dr. S. Amudha ASP/CINTEL · SRM IST
        </div>
    </div>
    """)

    sample_picker.change(fn=load_sample, inputs=sample_picker, outputs=log_input)

    submit_btn.click(
        fn=diagnose,
        inputs=log_input,
        outputs=[fault_output, confidence_output,
                 signals_output, signals_output, report_output]
    )

    app.load(fn=lambda: build_empty_report(), outputs=report_output)


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())