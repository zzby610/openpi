import os
from safetensors.torch import load_file

# 你的模型路径
ckpt_path = "checkpoints/train_lavida_libero/lavida_libero/30000/model.safetensors"

if not os.path.exists(ckpt_path):
    print(f"❌ 找不到文件: {ckpt_path}")
else:
    print(f"🔍 正在扫描模型: {ckpt_path} ...\n")
    # 只加载头信息，不占内存
    weights = load_file(ckpt_path, device="cpu")
    keys = list(weights.keys())
    
    # 定义“视觉组件”关键字
    vision_keywords = ["vision_tower", "mm_projector", "vision_resampler", "visual_projection"]
    action_keywords = ["action_head", "action_pred"]
    
    found_vision = [k for k in keys if any(v in k for v in vision_keywords)]
    found_actions = [k for k in keys if any(a in k for a in action_keywords)]
    
    print(f"📊 总权重项数量: {len(keys)}")
    print("-" * 50)
    
    if found_vision:
        print(f"✅ 发现视觉组件权重 ({len(found_vision)} 项)!")
        print(f"   示例 Key: {found_vision[:3]}")
    else:
        print("❌ 未发现视觉相关权重 (No vision_tower or mm_projector)!")
    
    if found_actions:
        print(f"✅ 发现动作头权重 ({len(found_actions)} 项)!")
        print(f"   示例 Key: {found_actions[:3]}")
    else:
        print("❓ 未发现明确的 action_head 权重 (可能包含在 transformer 内部)")

    # 打印前 10 个 Key 看看结构
    print("-" * 50)
    print("📋 前 10 个权重键名预览:")
    for k in keys[:10]:
        print(f"  - {k}")