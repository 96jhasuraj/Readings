
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED         = 42
N_CLASSES    = 3
N_PER_CLASS  = 150
HIDDEN_DIM   = 64
BATCH_SIZE   = 64
BOUNDARY_RES = 200
CLASS_COLORS = ['#D85A30', '#1D9E75', '#534AB7']
LIGHT_COLORS = ['#FAECE7', '#E1F5EE', '#EEEDFE']

torch.manual_seed(SEED)
np.random.seed(SEED)



def make_spiral(n_per_class=N_PER_CLASS, n_classes=N_CLASSES, noise=0.15):
    X, y = [], []
    for c in range(n_classes):
        t = np.linspace(0, 4 * np.pi, n_per_class)
        r = np.linspace(0.1, 1.0, n_per_class)
        offset = c * (2 * np.pi / n_classes)
        x0 = r * np.sin(t + offset) + np.random.randn(n_per_class) * noise
        x1 = r * np.cos(t + offset) + np.random.randn(n_per_class) * noise
        X.append(np.stack([x0, x1], axis=1))
        y.append(np.full(n_per_class, c))
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int64)

def _make_norm(fix: str, dim: int):
    if fix == 'batchnorm':  return nn.BatchNorm1d(dim)
    if fix == 'layernorm':  return nn.LayerNorm(dim)
    return None

class PlainBlock(nn.Module):
    def __init__(self, dim: int, fix: str = 'none'):
        super().__init__()
        self.norm   = _make_norm(fix, dim)
        self.linear = nn.Linear(dim, dim)
        self.act    = nn.ReLU()

    def forward(self, x):
        z = self.norm(x) if self.norm is not None else x
        return self.act(self.linear(z))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, fix: str = 'none'):
        super().__init__()
        self.norm   = _make_norm(fix, dim)   # applied to INPUT (pre-norm)
        self.linear = nn.Linear(dim, dim)
        self.act    = nn.ReLU()

    def forward(self, x):
        z = self.norm(x) if self.norm is not None else x  # normalise input
        z = self.act(self.linear(z))                       # transform
        return z + x                                       # add original x


def build_network(depth: int, residual: bool, fix: str = 'none',
                  in_dim: int = 2, hidden: int = HIDDEN_DIM,
                  out_dim: int = N_CLASSES) -> nn.Sequential:
    Block = ResidualBlock if residual else PlainBlock
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(depth):
        layers.append(Block(hidden, fix))
    layers.append(nn.Linear(hidden, out_dim))

    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
    return net

class GradientTracker:

    def __init__(self, model: nn.Module):
        self.norms  = []
        self._hooks = []
        self._buf   = []
        for _, module in model.named_modules():
            if isinstance(module, nn.Linear):
                h = module.register_full_backward_hook(self._hook)
                self._hooks.append(h)

    def _hook(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self._buf.append(grad_output[0].norm().item())

    def flush(self):
        self.norms = list(reversed(self._buf))
        self._buf.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()

def train_epoch(model, loader, optimizer, criterion, tracker,
                clip_norm: float = 1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        tracker.flush()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct    += (logits.argmax(1) == yb).sum().item()
        total      += len(xb)
    return total_loss / total, correct / total

@torch.no_grad()
def decision_boundary_grid(model, resolution=BOUNDARY_RES):
    xs = np.linspace(-1.4, 1.4, resolution, dtype=np.float32)
    ys = np.linspace(-1.4, 1.4, resolution, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    grid   = torch.from_numpy(np.c_[XX.ravel(), YY.ravel()])
    model.eval()
    probs  = torch.softmax(model(grid), dim=1).numpy()
    model.train()
    ZZ     = probs.argmax(1).reshape(XX.shape)
    return XX, YY, ZZ


def plot_boundary(ax, model, X, y, title):
    XX, YY, ZZ = decision_boundary_grid(model)
    cmap_bg = ListedColormap(LIGHT_COLORS)
    cmap_pt = ListedColormap(CLASS_COLORS)
    ax.contourf(XX, YY, ZZ, levels=[-0.5, 0.5, 1.5, 2.5],
                cmap=cmap_bg, alpha=0.65)
    ax.contour(XX, YY, ZZ, levels=[0.5, 1.5],
               colors='white', linewidths=1.0, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pt,
               s=16, edgecolors='white', linewidths=0.4, zorder=3)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.set_xticks([]); ax.set_yticks([])


def plot_gradient_heatmap(ax, plain_norms, res_norms):
    n      = max(len(plain_norms), len(res_norms), 1)
    pn     = (plain_norms + [0.0] * n)[:n]
    rn     = (res_norms   + [0.0] * n)[:n]
    y_pos  = np.arange(n)
    labels = [f'L{i+1}' for i in range(n)]
    max_v  = max(max(pn + rn, default=1e-9), 1e-9)

    ax.barh(y_pos - 0.2, pn, height=0.38,
            color='#D85A30', alpha=0.85, label='Plain')
    ax.barh(y_pos + 0.2, rn, height=0.38,
            color='#1D9E75', alpha=0.85, label='Residual')

    for i, v in enumerate(pn):
        if v < max_v * 0.01:
            ax.text(max_v * 0.005, y_pos[i] - 0.2, 'vanishing',
                    va='center', ha='left', fontsize=6.5, color='#993C1D')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Gradient norm', fontsize=9)
    ax.set_title('Gradient flow — per layer\n(shorter bars = vanishing)',
                 fontsize=10, fontweight='bold', pad=5)
    ax.legend(fontsize=8, loc='lower right')
    ax.invert_yaxis()

def run(depth=15, lr=0.005, epochs=500, fix='none',
        clip_norm=1.0, plot_every=50):
    print(f"\n{'='*65}")
    print(f"  Experiment | depth={depth}  lr={lr}  fix={fix}  clip={clip_norm}")
    print(f"{'='*65}\n")

    X_np, y_np = make_spiral()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    plain_net = build_network(depth, residual=False, fix=fix)
    res_net   = build_network(depth, residual=True,  fix=fix)

    plain_opt   = optim.Adam(plain_net.parameters(), lr=lr, weight_decay=1e-4)
    res_opt     = optim.Adam(res_net.parameters(),   lr=lr, weight_decay=1e-4)
    criterion   = nn.CrossEntropyLoss()

    plain_sched = optim.lr_scheduler.CosineAnnealingLR(plain_opt, T_max=epochs)
    res_sched   = optim.lr_scheduler.CosineAnnealingLR(res_opt,   T_max=epochs)

    plain_tracker = GradientTracker(plain_net)
    res_tracker   = GradientTracker(res_net)

    hist = {k: [] for k in ('plain_loss', 'res_loss', 'plain_acc', 'res_acc')}
    grad_hist = {'plain': [], 'res': []}
    epoch_ticks = []
    last_p, last_r = [], []

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    fig.suptitle(
        f'Plain vs Residual Network  |  depth={depth}  lr={lr}  '
        f'fix={fix}  clip_norm={clip_norm}',
        fontsize=13, fontweight='bold'
    )
    gs   = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.5, 1.1, 1.0])
    ax_bp = fig.add_subplot(gs[0, 0])
    ax_br = fig.add_subplot(gs[0, 1])
    ax_hm = fig.add_subplot(gs[0, 2])
    ax_lo = fig.add_subplot(gs[1, :2])
    ax_ac = fig.add_subplot(gs[1, 2])
    ax_gn = fig.add_subplot(gs[2, :])

    def refresh(ep):
        for ax, net, key, label in [
            (ax_bp, plain_net, 'plain', 'Plain net'),
            (ax_br, res_net,  'res',   'Residual net'),
        ]:
            ax.cla()
            plot_boundary(ax, net, X_np, y_np,
                          f'{label}  (ep {ep})\n'
                          f'acc={hist[key+"_acc"][-1]:.1%}  '
                          f'loss={hist[key+"_loss"][-1]:.3f}')

        ax_hm.cla()
        plot_gradient_heatmap(ax_hm, last_p, last_r)

        ax_lo.cla()
        ax_lo.plot(hist['plain_loss'], color='#D85A30', lw=1.8, label='Plain')
        ax_lo.plot(hist['res_loss'],   color='#1D9E75', lw=2.0, label='Residual')
        ax_lo.set_xlabel('Epoch', fontsize=9)
        ax_lo.set_ylabel('Cross-entropy loss', fontsize=9)
        ax_lo.set_title('Training loss', fontsize=11, fontweight='bold')
        ax_lo.legend(fontsize=9)

        ax_ac.cla()
        ax_ac.plot(hist['plain_acc'], color='#D85A30', lw=1.8, label='Plain')
        ax_ac.plot(hist['res_acc'],   color='#1D9E75', lw=2.0, label='Residual')
        ax_ac.set_ylim(0, 1.05)
        ax_ac.set_xlabel('Epoch', fontsize=9)
        ax_ac.set_ylabel('Accuracy', fontsize=9)
        ax_ac.set_title('Training accuracy', fontsize=11, fontweight='bold')
        ax_ac.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax_ac.legend(fontsize=9)

        ax_gn.cla()
        if grad_hist['plain']:
            ax_gn.plot(epoch_ticks, grad_hist['plain'], color='#D85A30',
                       lw=1.8, label='Plain — L1 grad norm')
            ax_gn.plot(epoch_ticks, grad_hist['res'],   color='#1D9E75',
                       lw=2.0, label='Residual — L1 grad norm')
            ax_gn.axhline(1e-5, color='gray', lw=0.8, ls='--',
                          label='Vanishing threshold (1e-5)')
        ax_gn.set_xlabel('Epoch', fontsize=9)
        ax_gn.set_ylabel('Layer-1 gradient norm', fontsize=9)
        ax_gn.set_title(
            'Early-layer gradient norm over time  '
            '(plain should be 10-100x smaller than residual)',
            fontsize=11, fontweight='bold')
        ax_gn.legend(fontsize=9)
        ax_gn.set_yscale('symlog', linthresh=1e-7)
        plt.pause(0.01)

    # ── Training loop
    plt.ion()
    for ep in range(1, epochs + 1):
        p_loss, p_acc = train_epoch(plain_net, loader, plain_opt,
                                    criterion, plain_tracker, clip_norm)
        r_loss, r_acc = train_epoch(res_net,   loader, res_opt,
                                    criterion, res_tracker,   clip_norm)
        plain_sched.step()
        res_sched.step()

        hist['plain_loss'].append(p_loss)
        hist['res_loss'].append(r_loss)
        hist['plain_acc'].append(p_acc)
        hist['res_acc'].append(r_acc)

        if plain_tracker.norms: last_p = plain_tracker.norms[:]
        if res_tracker.norms:   last_r = res_tracker.norms[:]

        if last_p:
            grad_hist['plain'].append(last_p[0])
            grad_hist['res'].append(last_r[0] if last_r else 0.0)
            epoch_ticks.append(ep)

        if ep % 50 == 0 or ep == 1:
            pg = last_p[0] if last_p else 0.0
            rg = last_r[0] if last_r else 0.0
            flag = '  *** VANISHING ***' if pg < 1e-5 else ''
            print(f"Ep {ep:>4d}/{epochs}  "
                  f"Plain  loss={p_loss:.4f} acc={p_acc:.1%}  |  "
                  f"Resid  loss={r_loss:.4f} acc={r_acc:.1%}  |  "
                  f"L1 grad  plain={pg:.2e}  resid={rg:.2e}{flag}")

        if plot_every > 0 and (ep % plot_every == 0 or ep == 1):
            refresh(ep)

    plt.ioff()
    refresh(epochs)

    # ── Summary
    pg = last_p[0] if last_p else 0.0
    rg = last_r[0] if last_r else 0.0
    ratio = rg / (pg + 1e-12)
    print(f"\n{'─'*65}")
    print("  Final results")
    print(f"{'─'*65}")
    print(f"  Plain    loss={hist['plain_loss'][-1]:.4f}  "
          f"acc={hist['plain_acc'][-1]:.1%}")
    print(f"  Residual loss={hist['res_loss'][-1]:.4f}  "
          f"acc={hist['res_acc'][-1]:.1%}")
    print(f"\n  L1 grad  plain={pg:.2e}  residual={rg:.2e}  "
          f"({ratio:.1f}x stronger in residual)")
    print(f"\n  Diagnosis checklist:")
    res_won = hist['res_acc'][-1] > hist['plain_acc'][-1]
    res_stable = hist['res_loss'][-1] < 1.5
    plain_vanish = pg < 1e-5
    print(f"    Residual > Plain acc?   {'YES' if res_won else 'NO  <- try depth>=15 or fix=layernorm'}")
    print(f"    Residual loss stable?   {'YES' if res_stable else 'NO  <- reduce LR or add layernorm'}")
    print(f"    Plain vanishing?        {'YES (confirmed)' if plain_vanish else 'Mild <- try depth>=15'}")
    print(f"{'─'*65}\n")

    plain_tracker.remove()
    res_tracker.remove()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Plain vs Residual network — spiral dataset (v2 debugged)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python residual_experiment.py                              # baseline
  python residual_experiment.py --depth 20                   # deeper
  python residual_experiment.py --fix layernorm              # best combo
  python residual_experiment.py --fix batchnorm --lr 0.002   # BN needs lower LR
  python residual_experiment.py --depth 20 --fix layernorm --epochs 800
  python residual_experiment.py --plot_every 0               # faster, final only
        """
    )
    p.add_argument('--depth',      type=int,   default=15)
    p.add_argument('--lr',         type=float, default=0.005)
    p.add_argument('--epochs',     type=int,   default=500)
    p.add_argument('--fix',        type=str,   default='none',
                   choices=['none', 'batchnorm', 'layernorm'])
    p.add_argument('--clip_norm',  type=float, default=1.0)
    p.add_argument('--plot_every', type=int,   default=50)
    args = p.parse_args()

    run(depth=args.depth, lr=args.lr, epochs=args.epochs,
        fix=args.fix, clip_norm=args.clip_norm, plot_every=args.plot_every)