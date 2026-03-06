'''
Daten zeigen drei Ebenen der Emergenz:
    (A) Primäre Emergenz: lokale Verdichtung → ein Punkt entsteht
    (B) Sekundäre Emergenz: mehrere Punkte koppeln → Cluster/Filamente entstehen
    (C) Tertiäre Emergenz: Cluster verbinden sich → Netzwerk entsteht

Das ist genau die Art von Emergenz, die wir in: 
    Ökosystemen
    neuronalen Netzen
    sozialen Netzwerken
    frühen kosmologischen Strukturen
finden.

Das Modell zeigt, dass aus instabilen, stochastischen Energiepunkten durch lokale Interaktion stabile, 
makroskopische Muster entstehen, die höherer Ordnung sind als ihre Bestandteile.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel, convolve

# --------------------------- Parameters (tweakable) ---------------------------
GRID_SIZE = 60
dt = 0.05
steps = 250
np.random.seed(2)

# Agent parameters (12 agents)
E0 = np.array([-3,-2,-1,-1,0,1,1,2,2,2,3,3], dtype=float)
n_agents = len(E0)
kappa = 0.3 * np.ones(n_agents)
mu = E0.copy()
sigma_noise = 0.05 * np.ones(n_agents)
alpha = 1.0
w = np.full((n_agents,n_agents), 0.0)
groups = [0,0,0,1,1,1,2,2,2,3,3,3]
for i in range(n_agents):
    for j in range(n_agents):
        if groups[i] == groups[j]:
            w[i,j] = 0.5
        elif abs(groups[i]-groups[j]) == 1:
            w[i,j] = 0.1
        else:
            w[i,j] = -0.2

# Kernel for Phi
sigma_K = 2.5

# Packet generation parameters
beta = 1.2
gamma = 0.8
theta_c = 0.7
mu_e = 1.0
sigma_e = 0.25
max_packets = 2000
move_noise = 0.5

# Packet decay
lambda0 = 0.12
e_star = 1.0

# Condensation thresholds
rho_th = 4
var_th = 0.8
s_sigma = 0.2
s_delta = 0.1
s_stable = 0.45

# --------------------------- Initialization ---------------------------
grid_shape = (GRID_SIZE, GRID_SIZE)
agent_positions = np.zeros((n_agents,2), dtype=float)
xs = np.linspace(5, GRID_SIZE-5, n_agents)
ys = np.full(n_agents, GRID_SIZE/2.0)
agent_positions[:,0] = xs
agent_positions[:,1] = ys

E = E0.copy()

# Packets represented by numpy arrays for speed
max_slots = max_packets
pkt_x = np.zeros(max_slots, dtype=float)
pkt_y = np.zeros(max_slots, dtype=float)
pkt_e = np.zeros(max_slots, dtype=float)
pkt_alive = np.zeros(max_slots, dtype=bool)
pkt_count = 0

stable_entities = []

# Diagnostics
phi_last = None
num_packets_history = []
num_stable_history = []

# Precompute gaussian stamp for agent contribution
stamp_radius = int(np.ceil(3*sigma_K))
xs_stamp = np.arange(-stamp_radius, stamp_radius+1)
xxs, yys = np.meshgrid(xs_stamp, xs_stamp)
stamp = np.exp(-(xxs**2+yys**2)/(2*sigma_K**2))
stamp = stamp / stamp.sum()
stamp_h = stamp.shape[0]

def build_phi(agent_positions, E, grid_shape, stamp):
    phi = np.zeros(grid_shape, dtype=float)
    h, w_grid = grid_shape
    r = stamp.shape[0]//2
    for i, pos in enumerate(agent_positions):
        cx = int(round(pos[0])); cy = int(round(pos[1]))
        if cx < -r or cy < -r or cx > w_grid+r or cy > h+r:
            continue
        ex = max(0, cx-r); ey = max(0, cy-r)
        sx = ex - (cx-r); sy = ey - (cy-r)
        tx = min(w_grid, cx+r+1); ty = min(h, cy+r+1)
        tx_len = tx - ex; ty_len = ty - ey
        phi[ey:ty, ex:tx] += stamp[sy:sy+ty_len, sx:sx+tx_len] * (E[i])
    return phi

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# --------------------------- Simulation loop ---------------------------
for t in range(steps):
    # Update agent energies
    diffs_mat = E.reshape((n_agents,1)) - E.reshape((1,n_agents))
    interactions = np.sum(w * np.tanh(alpha * (-diffs_mat)), axis=1)
    drift = -kappa*(E - mu)
    E = E + (drift + interactions) * dt + sigma_noise * np.sqrt(dt) * np.random.normal(size=n_agents)
    E = np.clip(E, -10, 10)
    
    # Build Phi
    phi = build_phi(agent_positions, E, grid_shape, stamp)
    phi_sm = gaussian_filter(phi, sigma=0.9)
    phi_last = phi_sm.copy()
    
    # Gradient norm
    gx = sobel(phi_sm, axis=1)
    gy = sobel(phi_sm, axis=0)
    grad_norm = np.sqrt(gx*gx + gy*gy)
    
    # Packet creation
    avg_agent_E = np.mean(E)
    sample_cells = 300
    ys_rand = np.random.randint(0, GRID_SIZE, size=sample_cells)
    xs_rand = np.random.randint(0, GRID_SIZE, size=sample_cells)
    for xi, yi in zip(xs_rand, ys_rand):
        z = beta * grad_norm[yi, xi] + gamma * avg_agent_E - theta_c
        p_create = sigmoid(z)
        if np.random.rand() < p_create * dt and pkt_count < max_slots:
            x_pos = xi + np.random.rand() - 0.5
            y_pos = yi + np.random.rand() - 0.5
            e_val = max(0.01, np.random.normal(mu_e, sigma_e))
            pkt_x[pkt_count] = x_pos
            pkt_y[pkt_count] = y_pos
            pkt_e[pkt_count] = e_val
            pkt_alive[pkt_count] = True
            pkt_count += 1
    
    # Move packets and decay
    if pkt_count > 0:
        alive_idx = np.where(pkt_alive[:pkt_count])[0]
        if alive_idx.size > 0:
            ix = np.clip(np.round(pkt_x[alive_idx]).astype(int), 0, GRID_SIZE-1)
            iy = np.clip(np.round(pkt_y[alive_idx]).astype(int), 0, GRID_SIZE-1)
            gvx = gx[iy, ix]; gvy = gy[iy, ix]
            norm = np.hypot(gvx, gvy) + 1e-9
            vx = gvx / norm; vy = gvy / norm
            pkt_x[alive_idx] += vx * dt * 2.0 + np.sqrt(dt) * np.random.normal(scale=move_noise, size=alive_idx.size)
            pkt_y[alive_idx] += vy * dt * 2.0 + np.sqrt(dt) * np.random.normal(scale=move_noise, size=alive_idx.size)
            pkt_x[alive_idx] = np.clip(pkt_x[alive_idx], 0, GRID_SIZE-1)
            pkt_y[alive_idx] = np.clip(pkt_y[alive_idx], 0, GRID_SIZE-1)
            lam = lambda0 * np.exp(-pkt_e[alive_idx] / e_star)
            dec = np.random.rand(alive_idx.size) < (lam * dt)
            pkt_alive[alive_idx[dec]] = False
    
    # Compact occasionally
    if pkt_count > 0 and (t % 25 == 0):
        alive_mask = pkt_alive[:pkt_count]
        alive_idx = np.where(alive_mask)[0]
        if alive_idx.size > 0:
            pkt_x[:alive_idx.size] = pkt_x[alive_idx]
            pkt_y[:alive_idx.size] = pkt_y[alive_idx]
            pkt_e[:alive_idx.size] = pkt_e[alive_idx]
            pkt_alive[:alive_idx.size] = True
            pkt_count = alive_idx.size
        else:
            pkt_count = 0
    
    # Condensation
    pkt_grid = np.zeros(grid_shape, dtype=int)
    if pkt_count > 0:
        ix = np.clip(np.round(pkt_x[:pkt_count]).astype(int), 0, GRID_SIZE-1)
        iy = np.clip(np.round(pkt_y[:pkt_count]).astype(int), 0, GRID_SIZE-1)
        for xpi, ypi in zip(ix, iy):
            pkt_grid[ypi, xpi] += 1

    kernel3 = np.ones((3,3))
    neigh_count = convolve(pkt_grid.astype(float), kernel3, mode='constant')
    energy_grid = np.zeros(grid_shape, dtype=float)
    energy_sq_grid = np.zeros(grid_shape, dtype=float)

    if pkt_count > 0:
        for idx in range(pkt_count):
            if not pkt_alive[idx]: continue
            xi = int(round(pkt_x[idx])); yi = int(round(pkt_y[idx]))
            energy_grid[yi, xi] += pkt_e[idx]
            energy_sq_grid[yi, xi] += pkt_e[idx]**2

    neigh_energy = convolve(energy_grid, kernel3, mode='constant')
    neigh_energy_sq = convolve(energy_sq_grid, kernel3, mode='constant')

    with np.errstate(divide='ignore', invalid='ignore'):
        neigh_mean = np.where(neigh_count>0, neigh_energy / neigh_count, 0.0)
        neigh_var = np.where(neigh_count>0, (neigh_energy_sq / neigh_count) - neigh_mean**2, 0.0)

    # --- HERE: Corrected D-matter creation logging ---
    candidates = np.argwhere((neigh_count >= rho_th) & (neigh_var < var_th))
    for (iy, ix) in candidates:
        sM = s_sigma * (neigh_count[iy,ix] - rho_th) - s_delta * neigh_var[iy,ix]
        if sM > s_stable:
            stable_entities.append({
                'x': ix,
                'y': iy,
                'stability': float(sM),
                'created': int(t)
            })

            print(f"[t={t}] D-Materie entstanden bei ({ix},{iy}) – Stabilität {sM:.3f}")

            # remove packets in area
            if pkt_count > 0:
                x0 = max(0, ix-1); x1 = min(GRID_SIZE-1, ix+1)
                y0 = max(0, iy-1); y1 = min(GRID_SIZE-1, iy+1)
                mask = (np.round(pkt_x[:pkt_count])>=x0) & (np.round(pkt_x[:pkt_count])<=x1) & \
                       (np.round(pkt_y[:pkt_count])>=y0) & (np.round(pkt_y[:pkt_count])<=y1) & pkt_alive[:pkt_count]
                pkt_alive[:pkt_count][mask] = False

    if pkt_count > 0 and t % 20 == 0:
        alive_mask = pkt_alive[:pkt_count]
        alive_idx = np.where(alive_mask)[0]
        if alive_idx.size > 0:
            pkt_x[:alive_idx.size] = pkt_x[alive_idx]
            pkt_y[:alive_idx.size] = pkt_y[alive_idx]
            pkt_e[:alive_idx.size] = pkt_e[alive_idx]
            pkt_alive[:alive_idx.size] = True
            pkt_count = alive_idx.size
        else:
            pkt_count = 0

    num_packets_history.append(int(pkt_count))
    num_stable_history.append(len(stable_entities))

    if len(stable_entities) > 80:
        break

# --------------------------- Plots ---------------------------
fig, axes = plt.subplots(1,3, figsize=(12,4))

axes[0].set_title("Phi field (smoothed)")
im0 = axes[0].imshow(phi_last, origin='lower', interpolation='nearest')
fig.colorbar(im0, ax=axes[0])

axes[1].set_title(f"Packets (n={pkt_count}) and stable entities (n={len(stable_entities)})")
if pkt_count>0:
    axes[1].scatter(pkt_x[:pkt_count], pkt_y[:pkt_count], s=6, alpha=0.6)

if stable_entities:
    xs_s = [s['x'] for s in stable_entities]
    ys_s = [s['y'] for s in stable_entities]
    axes[1].scatter(xs_s, ys_s, s=80, marker='X', color="black")

    # Add creation time labels
    for s in stable_entities:
        axes[1].text(s['x']+0.3, s['y']+0.3, f"{s['created']}", color="red", fontsize=8)

axes[1].set_xlim(0, GRID_SIZE-1)
axes[1].set_ylim(0, GRID_SIZE-1)

axes[2].set_title("Stable entities over time")
axes[2].plot(num_stable_history)
axes[2].set_xlabel("step")
axes[2].set_ylabel("num stable")

plt.tight_layout()
plt.show()

# --------------------------- Summary ---------------------------
summary = {
    'steps_run': t+1,
    'final_num_packets': int(pkt_count),
    'final_num_stable_entities': len(stable_entities),
    'creation_times': [s['created'] for s in stable_entities],
    'example_stable_entity': stable_entities[0] if stable_entities else None
}

summary
