"""Optional graph travel-time initialization, independently derived from dynamics.

This initialization is an explicit implementation enhancement. The original
manuscript specifies the PINN PDE objective but no viscosity-solution initializer.
"""

import itertools

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def intersects_box(start,end,lower,upper):
    delta=end-start
    with np.errstate(divide="ignore",invalid="ignore"):
        a=(np.asarray(lower)-start)/delta
        b=(np.asarray(upper)-start)/delta
    parallel=np.abs(delta)<1e-12
    entry=np.where(parallel,-np.inf,np.minimum(a,b)).max(axis=-1)
    exit=np.where(parallel,np.inf,np.maximum(a,b)).min(axis=-1)
    outside=(parallel & ((start<lower)|(start>upper))).any(axis=-1)
    return (~outside)&(entry<=exit)&(entry<1)&(exit>0)


def graph_travel_time(grid,rho):
    size=int(np.prod(grid.shape));ids=np.arange(size).reshape(grid.shape)
    wind=grid.problem.wind(torch.from_numpy(grid.points)).numpy()
    speed=grid.problem.speed(torch.from_numpy(rho)).numpy()
    rows,cols,costs=[],[],[]
    for offset in itertools.product((-1,0,1),repeat=3):
        if offset==(0,0,0):
            continue
        left=tuple(slice(max(0,-s), min(n,n-s)) for n,s in zip(grid.shape,offset))
        right=tuple(slice(max(0,s), min(n,n+s)) for n,s in zip(grid.shape,offset))
        allowed=~(grid.obstacle[left]|grid.obstacle[right])
        start,end=grid.points[left],grid.points[right]
        for box in grid.problem.obstacles:
            allowed &= ~intersects_box(start,end,box["lower"],box["upper"])
        displacement=np.asarray(offset)*grid.spacing
        length=float(np.linalg.norm(displacement));direction=displacement/length
        drift=0.5*(wind[left]+wind[right]);local_speed=0.5*(speed[left]+speed[right])
        along=(drift*direction).sum(axis=-1)
        radicand=local_speed**2-(drift**2).sum(axis=-1)+along**2
        if np.any(radicand[allowed]<=0):
            raise ValueError("Graph initializer encountered uncontrollable drift")
        ground_speed=along+np.sqrt(np.maximum(radicand,1e-30))
        rows.append(ids[left][allowed]);cols.append(ids[right][allowed]);costs.append((length/ground_speed)[allowed])
    graph=coo_matrix((np.concatenate(costs),(np.concatenate(rows),np.concatenate(cols))),shape=(size,size)).tocsr()
    values=dijkstra(graph.transpose().tocsr(),directed=True,indices=ids[grid.target],min_only=True)
    if not np.isfinite(values[grid.active.ravel()]).all():
        raise RuntimeError("Graph initialization found geometrically unreachable active cells")
    values=values.reshape(grid.shape);values[grid.obstacle]=0
    return values


def initialize_potential(model,grid,rho,settings):
    if settings.get("initialization","analytic") != "graph_pretrain":
        return {"method":"analytic distance/free-speed coefficient"}
    device=next(model.parameters()).device
    reference=graph_travel_time(grid,rho)
    locations=torch.tensor(grid.points[grid.active],dtype=torch.float32,device=device)
    targets=torch.tensor(reference[grid.active],dtype=torch.float32,device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings["learning_rate"])
    scale=max(float(targets.max()),1)
    for epoch in range(settings.get("pretrain_epochs",1500)):
        index=torch.randint(len(locations),(min(5000,len(locations)),),device=device)
        prediction=model(locations[index])
        loss=((prediction-targets[index])/scale).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    print(f"Graph initialization: loss={float(loss.detach()):.3g}; subsequent solve uses PINN residual",flush=True)
    return {"method":"graph-pretrained PINN","epochs":settings.get("pretrain_epochs",1500),"normalized_mse":float(loss.detach()),
            "disclosure":"Implementation enhancement; not described in manuscript; graph values used only for initialization"}
