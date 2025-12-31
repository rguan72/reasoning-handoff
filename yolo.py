# %%
import pandas as pd
qwen_log = pd.read_json("qwen_debug_log.jsonl", lines=True)
# %%
qwen_log[qwen_log["problem_id"] == "test/intermediate_algebra/345.json"]

# %%
