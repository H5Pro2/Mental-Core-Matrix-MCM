# mcm_robustness.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel, convolve
import math, time

# --------------------------- Parameters (unchanged) ---------------------------
GRID_SIZE = 60
dt = 0.05
steps = 250

E0 = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=float)
n_agents = len(E0)
kappa = 0.3 * np.ones(n_agents)
mu = E0.copy()
sigma_noise = 0.05 * np.ones(n_agents)
alpha = 1.0

w = np.full((n_agents,n_agents), 0.0)
groups = [1,1,1,   2,2,2,   0,   3,3,3,   4,4,4]
for i in range(n_agents):
    for j in range(n_agents):
        if groups[i] == groups[j]:
            w[i,j] = 0.5
        elif abs(groups[i]-groups[j]) == 1:
            w[i,j] = 0.1
        else:
            w[i,j] = -0.2

sigma_K = 2.5

beta = 1.2
gamma = 0.8
theta_c = 0.7
mu_e = 1.0
sigma_e = 0.25
max_packets = 2000
move_noise = 0.5

lambda0 = 0.12
e_star = 1.0

rho_th = 4
var_th = 0.8
s_sigma = 0.2
s_delta = 0.1
s_stable = 0.45

# --------------------------- Helper setup ---------------------------
grid_shape = (GRID_SIZE, GRID_SIZE)
agent_positions = np.zeros((n_agents,2), dtype=float)
xs = np.linspace(5, GRID_SIZE-5, n_agents)
ys = np.full(n_agents, GRID_SIZE/2.0)
agent_positions[:,0] = xs
agent_positions[:,1] = ys

stamp_radius = int(np.ceil(3*sigma_K))
xs_stamp = np.arange(-stamp_radius, stamp_radius+1)
xxs, yys = np.meshgrid(xs_stamp, xs_stamp)
stamp = np.exp(-(xxs**2+yys**2)/(2*sigma_K**2))
stamp = stamp / stamp.sum()

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

# --------------------------- Core simulation function ---------------------------
def run_sim(seed, return_trajectory=False):
    np.random.seed(seed)
    E = E0.copy()

    pkt_x = np.zeros(max_packets, dtype=float)
    pkt_y = np.zeros(max_packets, dtype=float)
    pkt_e = np.zeros(max_packets, dtype=float)
    pkt_alive = np.zeros(max_packets, dtype=bool)
    pkt_count = 0

    stable_entities = []
    phi_last = None

    num_packets_history = []
    num_stable_history = []

    for t in range(steps):

        diffs_mat = E.reshape((n_agents,1)) - E.reshape((1,n_agents))
        interactions = np.sum(w * np.tanh(alpha * (-diffs_mat)), axis=1)
        drift = -kappa*(E - mu)
        E = E + (drift + interactions) * dt + sigma_noise * np.sqrt(dt) * np.random.normal(size=n_agents)
        E = np.clip(E, -10, 10)

        phi = build_phi(agent_positions, E, grid_shape, stamp)
        phi_sm = gaussian_filter(phi, sigma=0.9)
        phi_last = phi_sm.copy()

        gx = sobel(phi_sm, axis=1)
        gy = sobel(phi_sm, axis=0)
        grad_norm = np.sqrt(gx*gx + gy*gy)

        avg_agent_E = np.mean(E)
        sample_cells = 300
        ys_rand = np.random.randint(0, GRID_SIZE, size=sample_cells)
        xs_rand = np.random.randint(0, GRID_SIZE, size=sample_cells)

        for xi, yi in zip(xs_rand, ys_rand):
            z = beta * grad_norm[yi, xi] + gamma * avg_agent_E - theta_c
            p_create = sigmoid(z)
            if np.random.rand() < p_create * dt and pkt_count < max_packets:
                pkt_x[pkt_count] = xi + np.random.rand() - 0.5
                pkt_y[pkt_count] = yi + np.random.rand() - 0.5
                pkt_e[pkt_count] = max(0.01, np.random.normal(mu_e, sigma_e))
                pkt_alive[pkt_count] = True
                pkt_count += 1

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
                xi = max(0, min(GRID_SIZE-1, xi))
                yi = max(0, min(GRID_SIZE-1, yi))
                energy_grid[yi, xi] += pkt_e[idx]
                energy_sq_grid[yi, xi] += pkt_e[idx]**2

        neigh_energy = convolve(energy_grid, kernel3, mode='constant')
        neigh_energy_sq = convolve(energy_sq_grid, kernel3, mode='constant')
        with np.errstate(divide='ignore', invalid='ignore'):
            neigh_mean = np.where(neigh_count>0, neigh_energy / neigh_count, 0.0)
            neigh_var = np.where(neigh_count>0, (neigh_energy_sq / neigh_count) - neigh_mean**2, 0.0)

        candidates = np.argwhere((neigh_count >= rho_th) & (neigh_var < var_th))
        for (iy, ix) in candidates:
            sM = s_sigma * (neigh_count[iy,ix] - rho_th) - s_delta * neigh_var[iy,ix]
            if sM > s_stable:
                stable_entities.append({'x': ix, 'y': iy, 'stability': float(sM), 'created': int(t)})
                x0 = max(0, ix-1); x1 = min(GRID_SIZE-1, ix+1)
                y0 = max(0, iy-1); y1 = min(GRID_SIZE-1, iy+1)
                if pkt_count > 0:
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

    summary = {
        'steps_run': t+1,
        'final_num_packets': int(pkt_count),
        'final_num_stable_entities': len(stable_entities),
        'stable_entities': stable_entities  # <--- wichtigste Änderung
    }
    if return_trajectory:
        return summary, (pkt_x.copy(), pkt_y.copy(), pkt_e.copy(), pkt_alive.copy(), num_packets_history, num_stable_history, phi_last)
    return summary

# --------------------------- Batch runs (Robustheitstest) ---------------------------
if __name__ == "__main__":
    runs = 100
    start = time.time()
    results = []
    for r in range(runs):
        seed = 1000 + r
        res = run_sim(seed)
        results.append(res)
    elapsed = time.time() - start

    num_success = sum(1 for r in results if r['final_num_stable_entities'] >= 1)
    fraction = num_success / runs
    final_counts = [r['final_num_stable_entities'] for r in results]

    print(f"Robustheitstest: runs={runs}, successes={num_success}, fraction={fraction:.3f}, elapsed={elapsed:.1f}s")
    print("Basic statistics for stable entity counts:")
    print(" mean:", np.mean(final_counts), " median:", np.median(final_counts), " std:", np.std(final_counts))

    plt.figure(figsize=(6,3))
    plt.hist(final_counts, bins=30)
    plt.xlabel("final_num_stable_entities"); plt.ylabel("count")
    plt.title("Distribution of stable entities across runs")
    plt.tight_layout()
    plt.show()

    # Representative run
    rep_seed = 1000
    summary_rep, traj = run_sim(rep_seed, return_trajectory=True)
    pkt_x, pkt_y, pkt_e, pkt_alive, num_packets_history, num_stable_history, phi_last = traj

    fig, axes = plt.subplots(1,2, figsize=(12,5))
    # --- Übergangszonen U1..U4 (visuell) ---
    # Gruppenindizes (so wie du sie definiert hast)
    g1 = [0,1,2]
    g2 = [3,4,5]
    n0 = [6]
    g3 = [7,8,9]
    g4 = [10,11,12]

    def group_center(idxs):
        return agent_positions[idxs, 0].mean()

    cx_g1 = group_center(g1)
    cx_g2 = group_center(g2)
    cx_n0 = group_center(n0)
    cx_g3 = group_center(g3)
    cx_g4 = group_center(g4)

    # Übergänge als Mittelpunkte zwischen den Gruppen:
    u_positions = [
        0.5*(cx_g1 + cx_g2),  # U1
        0.5*(cx_g2 + cx_n0),  # U2
        0.5*(cx_n0 + cx_g3),  # U3
        0.5*(cx_g3 + cx_g4)   # U4
    ]

    # Breite des markierten Übergangsbandes
    u_w = 0.8

    # In beiden Plots markieren
    for ax in axes:
        for ux in u_positions:
            ax.axvspan(ux - u_w, ux + u_w, alpha=0.18, color='white', linewidth=0)

    axes[0].set_title("Phi field (smoothed) representative run")
    im = axes[0].imshow(phi_last, origin='lower', interpolation='nearest')
    fig.colorbar(im, ax=axes[0])

    axes[1].set_title(
        f"Packets + Stable Clusters (alive={np.sum(pkt_alive)}, clusters={summary_rep['final_num_stable_entities']})"
    )
    axes[1].set_xlim(0, GRID_SIZE-1)
    axes[1].set_ylim(0, GRID_SIZE-1)

    alive_idx = np.where(pkt_alive[:len(pkt_alive)])[0]
    if alive_idx.size > 0:
        axes[1].scatter(pkt_x[alive_idx], pkt_y[alive_idx], s=6, alpha=0.25, color="gray")

    # --- REAL CLUSTERS (MCM Condensation Events) ---
    stable_entities = summary_rep["stable_entities"]
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ci, ent in enumerate(stable_entities):
        cx = ent["x"]
        cy = ent["y"]
        color = cluster_colors[ci % len(cluster_colors)]
        axes[1].scatter(cx, cy, s=80, color=color, edgecolors='black', linewidths=1.0, label=f"C{ci}")

    if len(stable_entities) > 0:
        axes[1].legend(
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            fontsize=7,
            ncol=1,
            borderaxespad=0.
    )

    plt.tight_layout()
    plt.show()
