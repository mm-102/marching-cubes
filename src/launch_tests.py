import subprocess
import numpy as np
import argparse
from tqdm import tqdm
import time
import statistics

def run_test(dim, exe_path, shape) -> list[float]:
    cmd = [
        exe_path,
        f"--dim={dim}",
        f"--shape={shape}",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        return [float(x) for x in stdout.strip().split(";")]
    else:
        raise Exception(stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_dim', type=int, default=10)
    parser.add_argument('--max_dim', type=int, default=200)
    parser.add_argument('--num_tests', type=int, default=40)
    parser.add_argument('--exe', type=str, default='./time_test')
    parser.add_argument('--out', type=str, default='out.csv')
    parser.add_argument('--rep', type=int, default=5)
    parser.add_argument('--shape', type=str, default="perlin")
    args = parser.parse_args()

    with open(args.out, "w") as f:
        f.write("dim^3;seq;omp;cuda_calc;cuda_all\n")

    dims = np.linspace(args.min_dim, args.max_dim, args.num_tests).astype(int)

    for i, dim in enumerate(tqdm(dims, desc=f"Shape: {args.shape}", dynamic_ncols=True)):
        results = []
        for r in range(args.rep):
            try:
                _, *out = run_test(dim, args.exe, args.shape)
                results.append(out)
            except Exception as e:
                print(f"\nError on dim {dim}, rep {r}: {e}")
                continue
            time.sleep(0.1)

        if len(results) < 3:
            print(f"\nToo few results for dim {dim}, skipping...")
            continue

        # Transpose to get per-metric lists (seq, omp, cuda_calc, cuda_all)
        seqs, omps, cudas, alls = zip(*results)

        def avg_wo_min_max(values):
            if len(values) <= 2:
                return round(statistics.mean(values))
            trimmed = sorted(values)[1:-1]
            return round(statistics.mean(trimmed))

        avg_seq = avg_wo_min_max(seqs) / 1000
        avg_omp = avg_wo_min_max(omps) / 1000
        avg_cuda = avg_wo_min_max(cudas) / 1000
        avg_all = avg_wo_min_max(alls) / 1000

        # Log to file
        with open(args.out, "a") as f:
            f.write(f"{dim};{avg_seq};{avg_omp};{avg_cuda};{avg_all}\n")

        # Update progress bar postfix
        tqdm.write(f"\033[K[Dim {dim:4}] Seq: {avg_seq:.3f} ms | OMP: {avg_omp:.3f} ms | CUDA: {avg_cuda:.3f} / {avg_all:.3f} ms")

    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, stopping...")