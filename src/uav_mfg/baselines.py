"""Independent numerical reference and monolithic neural comparison methods."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .experiment import environment_metadata, save_json, write_csv
from .problem import Problem
from .reference import fast_sweeping, upwind_gradient
from .transport import Grid, Transport
from .value import Potential, evaluate, gradient, sample_points


def save_fields(output, config, grid, phi, rho, velocity, residual, diagnostics, method):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/"config.yaml").open("w") as stream:
        yaml.safe_dump(config,stream,sort_keys=False)
    np.savez_compressed(output/"fields.npz",phi=phi,density=rho,velocity=velocity,eikonal_residual=residual,
                        source=grid.source,active=grid.active,target=grid.target,obstacle=grid.obstacle,
                        x=grid.axes[0],y=grid.axes[1],z=grid.axes[2])
    save_json(output/"diagnostics.json",diagnostics)
    metadata={"method":method,"environment":environment_metadata(),"diagnostics":diagnostics,
              "status":"converged" if diagnostics.get("picard_converged") else "completed_unconverged",
              "config":config,"seed":config.get("seed"),"duration_seconds":diagnostics["duration_seconds"]}
    native_build=Path("build/fsm_build.json")
    if method == "isotropic FSM-FVM" and native_build.exists():
        metadata["native_build"]=json.loads(native_build.read_text())
    save_json(output/"metadata.json",metadata)


def run_reference(config,output):
    started=time.perf_counter()
    torch.set_num_threads(config.get("cpu_threads",8))
    problem=Problem(config);grid=Grid(problem,tuple(config["grid"]))
    rho=np.zeros(grid.shape);history=[]
    converged=False
    for outer in range(config["coupling"]["max_iterations"]):
        speed=problem.speed(torch.from_numpy(rho)).numpy()
        phi,value_diagnostics=fast_sweeping(grid,speed,backend="openmp",threads=config.get("cpu_threads",8))
        grad=upwind_gradient(grid,phi)
        norm=np.sqrt((grad**2).sum(axis=-1)+config["training"]["epsilon"]**2)
        velocity=-speed[...,None]*grad/norm[...,None]
        transport=Transport(grid,velocity)
        proposal,inner=transport.solve(rho,**config["transport"])
        clipped=np.clip(proposal,0,problem.physics["rho_max"])
        accepted=(1-config["coupling"]["alpha"])*rho+config["coupling"]["alpha"]*clipped
        change=float(np.linalg.norm(accepted-rho)/max(np.linalg.norm(rho),1e-30))
        removed=float((proposal-clipped).sum()*grid.cell_volume)
        rho=accepted
        audit=transport.audit(rho)
        history.append({"outer":outer,"accepted_density_change":change,"projection_mass_removed":removed,**audit})
        if change<config["coupling"]["tolerance"] and inner["converged"] and removed<1e-10 and audit["relative_steady_balance"]<config["transport"]["balance_tolerance"]:
            converged=True;break
    speed=problem.speed(torch.from_numpy(rho)).numpy()
    phi,value_diagnostics=fast_sweeping(grid,speed,backend="openmp",threads=config.get("cpu_threads",8))
    grad=upwind_gradient(grid,phi);norm=np.sqrt((grad**2).sum(axis=-1)+config["training"]["epsilon"]**2)
    velocity=-speed[...,None]*grad/norm[...,None]
    residual=speed*norm-1
    diagnostics={**Transport(grid,velocity).audit(rho),"picard_converged":converged,"picard_iterations":len(history),
                 "eikonal_mean":float(np.abs(residual[grid.active]).mean()),"eikonal_p95":float(np.quantile(np.abs(residual[grid.active]),0.95)),
                 "duration_seconds":time.perf_counter()-started,"value_solver":value_diagnostics,
                 "total_projection_mass_removed":sum(row["projection_mass_removed"] for row in history)}
    save_fields(output,config,grid,phi,rho,velocity,residual,diagnostics,"isotropic FSM-FVM")
    write_csv(Path(output)/"picard.csv",history)
    return diagnostics


class Density(nn.Module):
    def __init__(self,problem,widths):
        super().__init__();self.problem=problem
        modules=[];size=3
        for width in widths:
            modules.extend((nn.Linear(size,width),nn.Tanh()));size=width
        modules.append(nn.Linear(size,1));self.network=nn.Sequential(*modules)
        nn.init.constant_(modules[-1].bias,-3)
        self.register_buffer("lower",torch.tensor(problem.bounds[:,0],dtype=torch.float32))
        self.register_buffer("extent",torch.tensor(np.diff(problem.bounds,axis=1).ravel(),dtype=torch.float32))

    def forward(self,x):
        rho=self.problem.physics["rho_max"]*torch.sigmoid(self.network(2*(x-self.lower)/self.extent-1).squeeze(-1))
        return rho


def neural_fields(potential,density,x,epsilon,create_graph=True):
    phi=potential(x);rho=density(x);grad=gradient(phi,x,create_graph=create_graph)
    wind=potential.problem.wind(x);speed=potential.problem.speed(rho)
    norm=(grad.square().sum(dim=-1)+epsilon**2).sqrt()
    velocity=wind-speed[:,None]*grad/norm[:,None]
    return phi,rho,velocity,speed*norm-(wind*grad).sum(dim=-1)-1


def outer_boundary(problem,count,device):
    lower=torch.tensor(problem.bounds[:,0],dtype=torch.float32,device=device)
    extent=torch.tensor(np.diff(problem.bounds,axis=1).ravel(),dtype=torch.float32,device=device)
    x=lower+torch.rand(count,3,device=device)*extent
    faces=torch.randint(6,(count,),device=device);normal=torch.zeros_like(x)
    for d in range(3):
        x[:,d]=torch.where(faces==2*d,lower[d],x[:,d]);x[:,d]=torch.where(faces==2*d+1,lower[d]+extent[d],x[:,d])
        normal[:,d]=torch.where(faces==2*d,-1.,normal[:,d]);normal[:,d]=torch.where(faces==2*d+1,1.,normal[:,d])
    return x.detach().requires_grad_(True),normal,faces


def run_monolithic(config,output,device="cuda",seed=42,epochs=2000):
    started=time.perf_counter();torch.manual_seed(seed);torch.set_num_threads(config.get("cpu_threads",8))
    config["seed"]=seed
    problem=Problem(config);grid=Grid(problem,tuple(config["grid"]))
    if problem.obstacles or problem.wind_config["type"] != "none" or problem.target.get("type","sphere") != "sphere":
        raise ValueError("Monolithic comparison currently requires zero wind, no obstacles and a spherical target")
    potential=Potential(problem,config["network"]["widths"]).to(device)
    density=Density(problem,config["network"]["widths"]).to(device)
    optimizer=torch.optim.Adam(list(potential.parameters())+list(density.parameters()),lr=config["training"]["learning_rate"])
    epsilon=config["training"]["epsilon"];history=[]
    for epoch in range(epochs):
        x=sample_points(problem,config["training"]["pde_points"],device=device,buffer=config["barrier"]["buffer"]).requires_grad_(True)
        phi,rho,velocity,eik=neural_fields(potential,density,x,epsilon)
        flux=rho[:,None]*velocity
        divergence=sum(torch.autograd.grad(flux[:,d].sum(),x,create_graph=True,retain_graph=True)[0][:,d] for d in range(3))
        continuity=divergence-problem.source(x)
        wall,normal,_=outer_boundary(problem,config["training"]["boundary_points"],device)
        _,wall_rho,wall_u,_=neural_fields(potential,density,wall,epsilon)
        leakage=wall_rho*(wall_u*normal).sum(dim=-1)
        loss=eik.square().mean()+continuity.square().mean()+0.1*leakage.square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Monolithic objective is nonfinite")
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if epoch%100==0 or epoch==epochs-1:
            history.append({"epoch":epoch,"loss":float(loss.detach()),"eikonal_mean":float(eik.detach().abs().mean()),
                            "continuity_mean":float(continuity.detach().abs().mean()),"boundary_flux_mse":float(leakage.detach().square().mean())})
    locations=torch.tensor(grid.points.reshape(-1,3),dtype=torch.float32,device=device)
    with torch.no_grad():
        rho=density(locations).cpu().numpy().reshape(grid.shape)
    phi,velocity,residual=evaluate(potential,grid,rho,epsilon)
    # Audit the actual neural flux at boundaries, before any FVM no-flux projection.
    boundary,normal,faces=outer_boundary(problem,12000,device)
    _,wall_rho,wall_u,_=neural_fields(potential,density,boundary,epsilon,create_graph=False)
    signed_flux=(wall_rho*(wall_u*normal).sum(dim=-1)).detach().cpu().numpy();face_ids=faces.cpu().numpy()
    outer_net,outer_abs=0.,0.
    length=np.diff(problem.bounds,axis=1).ravel()
    for face in range(6):
        area=np.prod(np.delete(length,face//2));values=signed_flux[face_ids==face]
        outer_net+=float(values.mean()*area);outer_abs+=float(np.abs(values).mean()*area)
    if problem.target.get("type","sphere") != "sphere":
        raise ValueError("Monolithic target flux audit currently requires a spherical target")
    direction=torch.randn(12000,3,device=device);direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
    center=torch.tensor(problem.target["center"],dtype=torch.float32,device=device)
    surface=(center+(problem.target["radius"]+1e-4)*direction).requires_grad_(True)
    _,sink_rho,sink_u,_=neural_fields(potential,density,surface,epsilon,create_graph=False)
    absorption=float((-sink_rho*(sink_u*direction).sum(dim=-1)).mean().detach())*4*np.pi*problem.target["radius"]**2
    source_rate=float(grid.source.sum()*grid.cell_volume)
    diagnostics={"source_rate":source_rate,"target_absorption":absorption,"absorption_over_source":absorption/source_rate,
                 "outer_leakage":outer_abs,"outer_net_flux":outer_net,
                 "relative_steady_balance":abs(source_rate-absorption-outer_net)/source_rate,
                 "eikonal_mean":float(np.abs(residual[grid.active]).mean()),"eikonal_p95":float(np.quantile(np.abs(residual[grid.active]),0.95)),
                 "rho_max":float(rho[grid.active].max()),"duration_seconds":time.perf_counter()-started,"optimization_epochs":epochs,
                 "audit": "Monte Carlo quadrature of actual neural flux, seed recorded; zero-wind/no-obstacle comparison",
                 "picard_converged":False}
    if problem.obstacles:
        raise ValueError("Baseline audit is scoped to the paper's no-obstacle comparison")
    save_fields(output,config,grid,phi,rho,velocity,residual,diagnostics,"monolithic PINN")
    write_csv(Path(output)/"training.csv",history)
    torch.save({"potential":potential.state_dict(),"density":density.state_dict(),"config":config,"seed":seed},Path(output)/"checkpoint.pt")
    return diagnostics
