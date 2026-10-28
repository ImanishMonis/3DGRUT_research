import argparse
import json
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Batch run render_ply.py from a JSON file.")
    parser.add_argument("--json_path", required=True, help="Path to the JSON file containing the array.")
    parser.add_argument("--start", type=int, default=0, help="Starting index in the JSON array.")
    parser.add_argument("--limit", type=int, default=None, help="Number of elements from the array to process.")
    parser.add_argument("--checkpoint", default="/cache/s49imoni/runs/tmp/align_t1-2708_110255/ckpt_last.pt", help="Path to the checkpoint file.")
    parser.add_argument("--out_dir", default="./output", help="Directory to save the rendered output.")
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Slice data based on start index and optional limit
    end_idx = args.start + args.limit if args.limit is not None else None
    data = data[args.start:end_idx]

    for item in data:
        qw, qx, qy, qz = item["rotation_qw_qx_qy_qz"]
        tx, ty, tz = item["translation_tx_ty_tz"]
        # Append .png so torchvision/PIL can resolve the file format extension
        image_name = f"{item['image_id']}.png"
        
        cmd = [
            "python", "render_ply.py",
            "--checkpoint", args.checkpoint,
            "--tx", str(tx),
            "--ty", str(ty),
            "--tz", str(tz),      ## front back 
            "--qw", str(qw),
            "--qx", str(qx),
            "--qy", str(qy),
            "--qz", str(qz),
            "--out_dir", args.out_dir,
            "--name", image_name
        ]
        
        print(f"Running for image_id {item['image_id']} ({item['image_path']})...")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()