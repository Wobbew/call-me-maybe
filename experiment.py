import time

from llm_sdk import Small_LLM_Model
model = Small_LLM_Model()

ids = model.encode('{"name": "fn_')[0].tolist()

# First call is warm-up, second call is the real timing
model.get_logits_from_input_ids(ids)

start = time.perf_counter()
logits = model.get_logits_from_input_ids(ids)
print(f"one call: {time.perf_counter() - start:.2f}s")

print("number of logits:", len(logits))

top5 = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
for i in top5:
    print(i, repr(model.decode([i])), round(logits[i], 2))