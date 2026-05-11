import torch

from research.engine.domain import SpectralDomain
from research.engine.basis import GHGSFDualDomainBasis
from research.engine.topology import generateTopology

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
domain = SpectralDomain(380.0, 830.0, 4096, device=device, dtype=torch.float64)

K, N = 8, 11
TOPOLOGY_NAMES = {0: "Uniform All Wide", 1: "Uniform All Narrow", 2: "Uniform First Wide"}

for topo_id in [0, 1, 2]:
    centers, wideIndices = generateTopology(topo_id, K, margin=0.0)
    basis = GHGSFDualDomainBasis(
        domain=domain,
        centers=centers,
        wideIndices=wideIndices,
        wideSigmaMin=9.5,  wideSigmaMax=11.5,  wideScaleType="linear",
        narrowSigmaMin=7.0, narrowSigmaMax=9.0, narrowScaleType="linear",
        order=N
    )

    G  = basis.m_gram
    ev = torch.linalg.eigvalsh(G)
    cond = (ev[-1] / ev[0].clamp(min=1e-30)).item()

    print(f"[{topo_id}] {TOPOLOGY_NAMES[topo_id]:<22}  cond(G) = {cond:.4e}  λ_min = {ev[0].item():.3e}  λ_max = {ev[-1].item():.3e}")

# ── New candidate: all 3 uniform topologies ────────────────────────────────────
# K=8  order=10  linear  wide=[6.5,8.5]  narrow=[7.5,8.0]
print()
print("── New candidate  (K=8  order=10  linear  wide=[6.5,8.5]  narrow=[7.5,8.0]) ─")
K_new, N_new = 8, 10
for topo_id in [0, 1, 2]:
    centers, wideIndices = generateTopology(topo_id, K_new, margin=0.0)
    basis_new = GHGSFDualDomainBasis(
        domain=domain,
        centers=centers,
        wideIndices=wideIndices,
        wideSigmaMin=6.5,  wideSigmaMax=8.5,  wideScaleType="linear",
        narrowSigmaMin=7.5, narrowSigmaMax=8.0, narrowScaleType="linear",
        order=N_new,
    )
    ev   = torch.linalg.eigvalsh(basis_new.m_gram)
    cond = (ev[-1] / ev[0].clamp(min=1e-30)).item()
    print(f"  [{topo_id}] {TOPOLOGY_NAMES[topo_id]:<22}  M={basis_new.m_M}  "
          f"SPD={ev[0].item()>0}  cond(G)={cond:.4e}  "
          f"log10={torch.log10(torch.tensor(cond)).item():.3f}  "
          f"λ_min={ev[0].item():.3e}  λ_max={ev[-1].item():.3e}")