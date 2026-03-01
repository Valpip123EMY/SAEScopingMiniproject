"""
Plot SAE feature firing rate distributions saved by find_firing_rates.py.

Usage:
    python scripts/plot_firing_rates.py
    python scripts/plot_firing_rates.py --cache-dir scripts/.cache --ignore-padding True
"""
from __future__ import annotations
import click
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from safetensors.torch import load_file


def load_distributions(cache_dir: Path, ignore_padding: bool) -> dict[tuple[str, str], torch.Tensor]:
    """Returns {(dataset_name, sae_id): distribution_tensor}"""
    subfolder = cache_dir / f"ignore_padding_{ignore_padding}"
    results = {}
    for dataset_dir in sorted(subfolder.iterdir()):
        for sae_dir in sorted(dataset_dir.glob("*/*")):  # layer/width/canonical -> two levels
            key = (dataset_dir.name, sae_dir.relative_to(dataset_dir).as_posix())
            dist_path = sae_dir / "distribution.safetensors"
            if dist_path.exists():
                results[key] = load_file(dist_path)["distribution"]
    return results


def plot_sorted_firing_rates(distributions: dict[tuple[str, str], torch.Tensor], out_dir: Path):
    """Sorted firing rate per dataset/SAE — useful for picking a pruning threshold."""
    datasets = sorted(set(d for d, _ in distributions))
    sae_ids = sorted(set(s for _, s in distributions))

    for sae_id in sae_ids:
        fig, ax = plt.subplots(figsize=(8, 4))
        for dataset in datasets:
            dist = distributions.get((dataset, sae_id))
            if dist is None:
                continue
            sorted_dist, _ = dist.sort(descending=True)
            ax.plot(sorted_dist.numpy(), label=dataset)
        ax.set_title(f"Sorted firing rates — {sae_id}")
        ax.set_xlabel("Feature rank")
        ax.set_ylabel("Firing rate fraction")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"sorted__{sae_id.replace('/', '--')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_cumulative(distributions: dict[tuple[str, str], torch.Tensor], out_dir: Path):
    """Cumulative firing rate — shows how many features capture X% of total activity."""
    datasets = sorted(set(d for d, _ in distributions))
    sae_ids = sorted(set(s for _, s in distributions))

    for sae_id in sae_ids:
        fig, ax = plt.subplots(figsize=(8, 4))
        for dataset in datasets:
            dist = distributions.get((dataset, sae_id))
            if dist is None:
                continue
            sorted_dist, _ = dist.sort(descending=True)
            cumsum = sorted_dist.cumsum(dim=0).numpy()
            ax.plot(cumsum, label=dataset)
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90%")
        ax.axhline(0.99, color="black", linestyle="--", linewidth=0.8, label="99%")
        ax.set_title(f"Cumulative firing rate — {sae_id}")
        ax.set_xlabel("Top-k features")
        ax.set_ylabel("Fraction of total activity")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"cumulative__{sae_id.replace('/', '--')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_cross_dataset_overlap(distributions: dict[tuple[str, str], torch.Tensor], out_dir: Path, top_k: int = 500):
    """For each SAE, show which top-k features are shared vs domain-specific across datasets."""
    sae_ids = sorted(set(s for _, s in distributions))
    datasets = sorted(set(d for d, _ in distributions))

    for sae_id in sae_ids:
        top_k_sets = {}
        for dataset in datasets:
            dist = distributions.get((dataset, sae_id))
            if dist is None:
                continue
            _, indices = dist.sort(descending=True)
            top_k_sets[dataset] = set(indices[:top_k].tolist())

        if len(top_k_sets) < 2:
            continue

        # Overlap matrix
        labels = list(top_k_sets.keys())
        n = len(labels)
        matrix = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                matrix[i][j] = len(top_k_sets[a] & top_k_sets[b]) / top_k

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label=f"Jaccard overlap (top-{top_k})")
        ax.set_title(f"Top-{top_k} feature overlap — {sae_id}")
        fig.tight_layout()
        path = out_dir / f"overlap_top{top_k}__{sae_id.replace('/', '--')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


@click.command()
@click.option("--cache-dir", "-c", type=click.Path(path_type=Path), default=Path(__file__).parent / ".cache")
@click.option("--ignore-padding", "-i", type=str, default="True")
@click.option("--top-k", "-k", type=int, default=500, help="Top-k features for overlap plot")
def cli(cache_dir: Path, ignore_padding: str, top_k: int):
    ignore_padding_bool = ignore_padding.lower().strip() == "true"
    out_dir = cache_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading distributions from {cache_dir} (ignore_padding={ignore_padding_bool})...")
    distributions = load_distributions(cache_dir, ignore_padding_bool)
    if not distributions:
        print("No distributions found. Run find_firing_rates.py first.")
        return
    print(f"Loaded {len(distributions)} distributions.")

    plot_sorted_firing_rates(distributions, out_dir)
    plot_cumulative(distributions, out_dir)
    plot_cross_dataset_overlap(distributions, out_dir, top_k=top_k)
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    cli()
