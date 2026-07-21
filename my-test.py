import json

data = ["りんご", "バナナ", "みかん"]
dict = {"fruits": data}
json_str = json.dumps(dict, ensure_ascii=False, indent=2)
print(json_str)
