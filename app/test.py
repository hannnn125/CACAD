import yaml 
from CACAD_32B import CACAD_32B

with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

bot = CACAD_32B(config)

for k, v in bot.prompts.items():
    print(k, "len=", len(v), "preview=", repr(v[:80]))

# 2) 실제 조합 결과
print("=== category ===")
print(bot._load_prompt("방임", "category", pred_cluster=2, num_examples=2))
# print("=== follow ===")
# print(bot._load_prompt("방임", "follow"))
# print("=== token_prediction ===")
# print(bot._load_prompt("방임", "token_prediction"))