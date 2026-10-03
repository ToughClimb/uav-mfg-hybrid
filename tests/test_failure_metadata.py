import copy
import json

import torch
import pytest

from uav_mfg.cli import load_settings
from uav_mfg.experiment import run_hybrid
from uav_mfg.problem import Problem
from uav_mfg.transport import Grid
from uav_mfg.value import Potential


def test_unresolved_target_is_recorded_as_setup_failure(tmp_path):
    config=load_settings("configs/verified_demo.yaml")
    config["grid"]=[10,10,4]
    config["target"]["center"]=[300,300,300]
    with pytest.raises(ValueError,match="absorbing target"):
        run_hybrid(config,tmp_path,device="cpu")
    metadata=json.loads((tmp_path/"metadata.json").read_text())
    assert metadata["status"]=="failed"
    assert "absorbing target" in metadata["error"]


def test_resume_rejects_changed_source_and_records_failure(tmp_path):
    original=load_settings("configs/verified_demo.yaml")
    original["grid"]=[10,10,4]
    original["network"]["widths"]=[4]
    grid=Grid(Problem(original),tuple(original["grid"]))
    model=Potential(grid.problem,[4])
    optimizer=torch.optim.Adam(model.parameters())
    checkpoint=tmp_path/"input.pt"
    torch.save({"grid":list(grid.shape),"config":original,"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),"density":torch.zeros(grid.shape),"outer":0},checkpoint)
    changed=copy.deepcopy(original)
    changed["source"]["rate"]*=2
    output=tmp_path/"failed_resume"
    with pytest.raises(ValueError,match="source, wind, physics"):
        run_hybrid(changed,output,device="cpu",resume=checkpoint)
    assert json.loads((output/"metadata.json").read_text())["status"]=="failed"
