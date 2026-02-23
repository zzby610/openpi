import json

try:
    with open('/data/datasets/biyuz/libero_lerobot/meta/info.json', 'r') as f:
        info = json.load(f)
        
    print("\n" + "="*50)
    print("🚨 你的真实数据列名如下：")
    for k in info.get('features', {}).keys():
        print(" -", k)
    print("="*50 + "\n")
except Exception as e:
    print("读取失败，错误信息:", e)
