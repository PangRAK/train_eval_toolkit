import csv
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "01_train"
    / "LODO_training_v0_1.py"
)


def load_lodo_training_module():
    spec = importlib.util.spec_from_file_location(
        "lodo_training_v0_1_test_module", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_validation_metric_rows():
    module = load_lodo_training_module()
    log_history = [
        {"loss": 1.2, "learning_rate": 4e-5, "epoch": 0.5, "step": 10},
        {
            "eval_loss": 0.8,
            "eval_category_macro_f1": 0.72,
            "eval_category_accuracy": 0.75,
            "eval_category_parse_rate": 0.95,
            "epoch": 1.0,
            "step": 20,
        },
    ]

    rows = module.extract_validation_metric_rows(log_history)

    assert rows == [
        {
            "step": 20,
            "epoch": 1.0,
            "eval_loss": 0.8,
            "eval_category_macro_f1": 0.72,
            "eval_category_accuracy": 0.75,
            "eval_category_parse_rate": 0.95,
        }
    ]


def test_export_training_metric_artifacts(tmp_path):
    module = load_lodo_training_module()
    output_dir = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts"
    output_dir.mkdir()

    trainer_state = {
        "global_step": 40,
        "log_history": [
            {"loss": 1.2, "learning_rate": 4e-5, "epoch": 0.5, "step": 10},
            {
                "eval_loss": 0.8,
                "eval_category_macro_f1": 0.72,
                "eval_category_accuracy": 0.75,
                "eval_category_parse_rate": 0.95,
                "eval_runtime": 12.3,
                "epoch": 1.0,
                "step": 20,
            },
            {
                "eval_loss": 0.6,
                "eval_category_macro_f1": 0.81,
                "eval_category_accuracy": 0.84,
                "eval_category_parse_rate": 0.97,
                "eval_runtime": 11.2,
                "epoch": 2.0,
                "step": 40,
            },
        ],
    }
    (output_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    module.export_training_metric_artifacts(output_dir, artifact_dir)

    export_dir = artifact_dir / "training_metrics"
    assert (export_dir / "trainer_state.json").exists()
    assert (export_dir / "trainer_log_history.jsonl").exists()
    assert (export_dir / "validation_metrics_history.jsonl").exists()
    assert (export_dir / "validation_metrics_history.csv").exists()
    assert (export_dir / "validation_metrics_history.png").exists()
    assert (export_dir / "validation_metrics_history.png").stat().st_size > 0
    assert (export_dir / "validation_metrics_summary.json").exists()

    csv_path = export_dir / "validation_metrics_history.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["step"] == "20"
    assert rows[1]["eval_category_macro_f1"] == "0.81"

    summary = json.loads(
        (export_dir / "validation_metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["validation_eval_count"] == 2
    assert "eval_loss" in summary["metrics"]
