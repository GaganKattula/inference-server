import matplotlib.pyplot as plt
import json

filepath = "/Users/gagan/Documents/studysessions/appliedlearning/inference-server/bench/results.json"
with open(filepath) as f:
    results = json.load(f)

x = [float(k) for k in results["sb"].keys()]

ttft_sb = [v[0] for v in results["sb"].values()]
ttft_cbc = [v[0] for v in results["cbc"].values()]
ttft_cbp = [v[0] for v in results["cbp"].values()]

tpot_sb = [v[1] for v in results["sb"].values()]
tpot_cbc = [v[1] for v in results["cbc"].values()]
tpot_cbp = [v[1] for v in results["cbp"].values()]





fig, axes = plt.subplots(1, 2, figsize=(12, 5))   # one figure, two subplots

ax = axes[0]                      # grab a subplot

ax.plot(x, ttft_sb)       # draw a line
ax.plot(x, ttft_cbc)     
ax.plot(x, ttft_cbp)     
ax.set_xlabel("lam")
ax.set_ylabel("seconds")
ax.set_title("TTFT plot")

ax = axes[1] 

ax.plot(x, tpot_sb, label="Static batching")       # draw a line
ax.plot(x, tpot_cbc, label="Continuous Batching + Contiguous Cache")     
ax.plot(x, tpot_cbp, label="Continuous Batching + Paged Cache")     
ax.set_xlabel("lam")
ax.set_ylabel("seconds")
ax.set_title("TPOT plot")

fig.legend()
plt.tight_layout()
plt.savefig("plots/bench.png")