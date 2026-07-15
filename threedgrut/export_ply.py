import torch
from pathlib import Path

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.export.formats.ply import PLYExporter  # adjust import if needed


def export_ply(ckpt_path, out_path="gaussians.ply"):
    # Load checkpoint
    ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False
)

    # Load config + model
    conf = ckpt["config"]
    model = MixtureOfGaussians(conf)

    # Restore Gaussians
    model.init_from_checkpoint(ckpt, setup_optimizer=False)

    # Export using official exporter
    exporter = PLYExporter()

    exporter.export(
        model=model,
        output_path=Path(out_path),
        conf=conf,
    )

    print(f"✅ Saved PLY to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="gaussians.ply")
    args = parser.parse_args()

    export_ply(args.ckpt, args.out)