import json
import re
import regex
from prompt import test_openai_api


demoname = "ml100k/cluster_prompt.json"
prefixs = ""
with open(demoname, "r", encoding='utf-8') as f:
    data = json.load(f)
    for i in data["demo"]:
        prefixs += "viewing history: " + i["history"] + "\n"
        prefixs += "Please analyze the user's preferences in 100 words based on the viewing history.\n"
        prefixs += i["preference"] + "\n"


def contains_non_ascii(s: str):
    return bool(regex.search(r'[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]', s))

def fix_preferences(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0

    for i, item in enumerate(data):
        pref = item.get("preference", "")

        if not pref:
            continue

        if contains_non_ascii(pref):
            print("\n===== 检测到非 ASCII preference =====")
            print(f"索引 {i} 的原始 preference：\n{pref}\n")

            prompt = (
                "Viewing history:\n"
                f"{item.get('history')}\n\n"
                "\nPlease analyze the user's preferences in 100 words based on the viewing history."
            )

            new_pref = test_openai_api(prefixs + prompt)
            print(">>> 修复后的 preference：")
            print(new_pref, "\n")

            data[i]["preference"] = new_pref

            with open(filename, "w", encoding="utf-8") as fw:
                json.dump(data, fw, ensure_ascii=False)

            count += 1

    print("\n=== 处理完成 ===")
    print(f"\n共修复{count}个preference")

    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False)


if __name__ == "__main__":
    fix_preferences("ml100k/test.json")
    fix_preferences("ml100k/train.json")
    fix_preferences("ml100k/val.json")

