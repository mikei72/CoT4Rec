"""
生成初始的train、val、test
"""
import json
from openai import OpenAI

# use your api_key here
client = OpenAI(
    api_key="sk-htshishgcvtvqwqnzcglxxwjiejjjzrotcyzetyrkxhmeamw",
    base_url="https://api.siliconflow.cn/v1",
)


# TODO: The 'openai.base_url' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(base_url="https://api.chatanywhere.tech/v1")'
# openai.base_url = "https://api.chatanywhere.tech/v1"
# MODEL_NAME = "gpt-4"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "gpt-4o"
# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def test_openai_api(question):
    sign = True
    while sign:
        try:
            rsp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": question},
                ],
            )
            sign = False
        except:
            sign = True
    return rsp.choices[0].message.content


if __name__ == "__main__":
    demo_file = 'ml100k/cluster_prompt.json'

    with open(demo_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    demos = data.get("demo", [])
    print(f"Loaded {len(demos)} demos")

    for i, item in enumerate(demos):
        history = item.get("history", "")

        prompt = (
                "viewing history: "
                + item.get("history")
                + "\nPlease analyze the user's preferences in 100 words based on the viewing history."
        )

        preference = test_openai_api(prompt)
        item["preference"] = preference.strip()

    # 保存
    save_file = demo_file
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump({"demo": demos}, f, ensure_ascii=False, indent=4)

    print(f"✅ Preferences filled and saved to: {save_file}")
