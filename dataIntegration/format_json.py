import os
import json


key_list = ["history", "result", "front", "recommendation", "preference"]


def format_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_lines = ["["]

    for idx, item in enumerate(data):
        formatted_lines.append("    {")
        for key_idx, key in enumerate(key_list):
            if key in item:
                value = item[key]
                if isinstance(value, list):
                    value_str = json.dumps(value, ensure_ascii=False, separators=(',', ': '))
                else:
                    value_str = json.dumps(value, ensure_ascii=False)
                comma = "," if key_idx < (len(key_list) - 1) and any(k in item for k in key_list[key_idx+1:]) else ""
                formatted_lines.append(f'        "{key}": {value_str}{comma}')
        formatted_lines.append("    }" + ("," if idx < len(data) - 1 else ""))
    formatted_lines.append("]")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(formatted_lines))

    print(f"✅ 已格式化文件：{os.path.basename(filepath)}")


format_json_file('ml100k/train.json')
format_json_file('ml100k/test.json')
format_json_file('ml100k/val.json')
