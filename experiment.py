# %%
import pandas as pd
ordered = pd.read_json("thought_sampling_out/ordered_results.jsonl", lines=True)
all_r = pd.read_json("thought_sampling_out/resample_results.jsonl", lines=True)
# %%
ordered.iloc[20:30]["off_policy_histogram"]
# %%
all_r[(all_r["cot_sample_id"] == 223) & (all_r["fraction"] == 0.75)]

# %%
all_r.iloc[6710]["completion"]
# %%
all_r.iloc[13010]["completion"]

# %%
ordered.tail()
# %%
