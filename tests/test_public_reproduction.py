"""Publication must work without private manuscript files or NVIDIA tools."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from uav_mfg.cli import load_settings
from uav_mfg.experiment import environment_metadata, resume_projection_mass, run_hybrid


def test_cpu_request_uses_cpu_transport_and_saves_real_status(tmp_path, monkeypatch):
    config = load_settings("configs/verified_demo.yaml")
    config["name"] = "cpu_smoke"
    config["grid"] = [10, 10, 4]
    config["network"]["widths"] = [8]
    config["training"].update(initialization=None, initial_epochs=1, subsequent_epochs=1,
                              pde_points=32, boundary_points=8, causality_points=8)
    config["transport"].update(max_steps=10)
    config["coupling"]["max_iterations"] = 1
    monkeypatch.setattr("uav_mfg.experiment.environment_metadata", lambda: {"test": True})
    result = run_hybrid(config, tmp_path, device="cpu", seed=42)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert config["transport"]["backend"] == "cuda"  # Caller configuration is preserved.
    assert metadata["config"]["transport"]["backend"] == "cpu"
    assert metadata["status"] == "completed_unconverged"
    assert not result["picard_converged"]
    assert (tmp_path / "fields.npz").is_file()
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert checkpoint["cumulative_projection_mass_removed"] == result["total_projection_mass_removed"]


def test_missing_optional_nvidia_tools_are_recorded(monkeypatch):
    monkeypatch.setattr("uav_mfg.experiment.shutil.which", lambda command: None)
    metadata = environment_metadata()
    assert metadata["nvidia_smi"]["available"] is False
    assert "not installed" in metadata["nvidia_smi"]["reason"]
    assert metadata["source_sha256"]


def test_previous_experiment_is_not_overwritten(tmp_path):
    original = '{"status":"completed_unconverged","important":"retain this record"}'
    (tmp_path / "metadata.json").write_text(original)
    with pytest.raises(FileExistsError, match="fresh output directory"):
        run_hybrid(load_settings("configs/verified_demo.yaml"), tmp_path, device="cpu")
    assert (tmp_path / "metadata.json").read_text() == original


def test_continuations_keep_cumulative_projection_mass(tmp_path):
    earlier = tmp_path / "earlier.pt"
    torch.save({"cumulative_projection_mass_removed": 0.125}, earlier)
    continued = tmp_path / "continued"
    continued.mkdir()
    checkpoint = {"outer": 7}
    (continued / "metadata.json").write_text(json.dumps({"resume": str(earlier)}))
    (continued / "picard.csv").write_text("outer,accepted_projection_mass_removed\n6,0.02\n7,0.03\n")
    assert resume_projection_mass(checkpoint, continued / "checkpoint.pt") == pytest.approx(0.175)
    assert resume_projection_mass({"cumulative_projection_mass_removed": 0.175}, earlier) == pytest.approx(0.175)
    with pytest.raises(ValueError, match="adjacent picard.csv"):
        resume_projection_mass({"outer": 0}, tmp_path / "incomplete.pt")


def test_manuscript_reference_is_optional_and_explicit(tmp_path):
    spec = importlib.util.spec_from_file_location("paper_report", Path("scripts/build_paper_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comparison = module.copy_reference_paper(None, tmp_path)
    assert comparison["enabled"] is False
    assert module.reference_section(comparison, ("fig3", "fig4"), "comparison") == ""
    assert not (tmp_path / "reference").exists()
    with pytest.raises(FileNotFoundError):
        module.copy_reference_paper(tmp_path / "absent", tmp_path)
