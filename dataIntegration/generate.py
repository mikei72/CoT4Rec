import json
import time
from prompt import test_openai_api


def generate_cot(filename, demoname=None):
    prefixs = ""

    if demoname:
        with open(demoname, "r", encoding='utf-8') as f:
            data = json.load(f)
            for i in data["demo"]:
                prefixs += "viewing history: " + i["history"] + "\n"
                prefixs += "Please analyze the user's preferences in 100 words based on the viewing history.\n"
                prefixs += i["preference"] + "\n"

    fp = None
    with open(filename, "r", encoding='utf-8') as f:
        fp = json.load(f)

    total_count = len(fp)
    print(f"共有 {total_count} 条")

    start_time = time.time()
    batch_start = start_time

    for i, v in enumerate(fp[:]):
        if v.get("preference"):
            continue

        prompt = (
            "viewing history: "
            + v.get("history")
            + "\nPlease analyze the user's preferences in 100 words based on the viewing history."
        )

        fp[i]["preference"] = test_openai_api(prefixs + prompt)

        if (i + 1) % 10 == 0 or i == len(fp) - 1:
            batch_end = time.time()
            batch_time = batch_end - batch_start

            elapsed = batch_end - start_time
            avg_time_per_item = elapsed / (i + 1)
            remaining_items = total_count - (i + 1)
            estimated_remaining = avg_time_per_item * remaining_items

            print(filename, "已生成数量：", i + 1)
            print(f"  运行时间（本批）：{batch_time:.2f}s, 已运行：{elapsed:.2f}s, 预计剩余：{estimated_remaining:.2f}s")

            with open(filename, "w") as wf:
                json.dump(fp, wf)

            batch_start = time.time()


# generate_cot("ml100k/test.json", "ml100k/cluster_prompt.json")
#generate_cot("ml100k/val.json", "ml100k/cluster_prompt.json")
generate_cot("ml100k/train.json", "ml100k/cluster_prompt.json")

