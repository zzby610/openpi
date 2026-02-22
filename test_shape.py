import sys
import os
sys.path.append(os.path.abspath("src"))  # 把 src 目录加入 Python 搜索路径
sys.path.append("/export/ra/zoubiyu/repo/LaMDA/LaMDA")
import torch
import logging
import torchvision.transforms.functional as TF

# 配置日志
logging.basicConfig(level=logging.INFO)

# 导入 LaMDA 相关的类和 LeRobot 数据集
try:
    from openpi.models.lamda_config import LaMDAConfig
    from openpi.models_pytorch.lamda_pytorch import LaMDAPytorch, LaMDAWithExpertModel
    from openpi.models.tokenizer import LaMDATokenizer
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    logging.info("成功导入 LaMDA 与 LeRobot 模块！")
except ImportError as e:
    logging.error(f"导入失败，请检查文件路径和环境: {e}")
    logging.error("👉 如果是找不到 lerobot，请在终端运行: pip install lerobot")
    exit(1)

def test_lamda_shapes():
    logging.info("=== 开始实弹测试：LaMDA + 真实 LIBERO 数据 ===")
    
    # 1. 设定测试参数
    batch_size = 1  # 真实测试先用单条数据跑通
    seq_len = 32    # 假装用户的文本指令长度
    hidden_dim = 4096
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 
    
    logging.info(f"使用设备: {device}, 数据类型: {dtype}")

    # ==========================================
    # 2. 初始化模型
    # ==========================================
    logging.info("正在初始化模型 (这可能需要一点时间)...")
    try:
        from types import SimpleNamespace
        dummy_expert_config = SimpleNamespace(
            head_dim=256,
            width=4096,      # 对齐 LLaDA 的 4096 维
            mlp_dim=16384, 
            num_heads=16,
            depth=4,         
            num_kv_heads=16
        )

        model = LaMDAWithExpertModel(
            lamda_model_path="/data/models/biyuz/hf_home/models/LLaDA-8B-Base",
            action_expert_config=dummy_expert_config,
            precision="bfloat16"
        ).to(device, dtype=dtype)
        
        # ⚠️ 这里非常重要：开启训练模式，因为我们要测反向传播
        model.train() 
        logging.info("模型初始化成功！")
        
        # --- 核心修改：动态对齐图片尺寸 ---
        # 自动从模型获取 patch_size (比如 16)
        p = model.vlm.lamda.config.vit_config.patch_size
        # 强制将测试图片调整为 p 的整数倍 (比如 16 * 37 = 592)
        image_size = p * 37 
        logging.info(f"检测到模型 patch_size 为 {p}，自动调整真实图片尺寸为 {image_size}x{image_size}")
        
    except Exception as e:
        import traceback
        logging.error(f"模型初始化失败:")
        traceback.print_exc()
        return

    # ==========================================
    # 3. 加载真实 LIBERO 数据
    # ==========================================
    logging.info("正在加载 LeRobot 真实数据集...")
    try:
        # 指向你刚才下载的文件夹路径
        dataset_path = "/data/datasets/biyuz/libero_lerobot" 
        dataset = LeRobotDataset(dataset_path)
        
        # 取出第 0 帧的数据看看
        sample = dataset[0]
        
        # 提取真实图片 (LeRobot 默认是 [C, H, W] 的 float32)
        img_tensor = sample["observation.images.image"] 
        # 拉伸到刚才计算好的 image_size (592x592)
        img_resized = TF.resize(img_tensor, [image_size, image_size], antialias=True)
        # 增加 Batch 维度并转到显卡: [1, 3, 592, 592]
        real_images = img_resized.unsqueeze(0).to(device, dtype=dtype)
        
        # 提取真实动作 (7维: 坐标+旋转+夹爪) -> [1, 7]
        real_actions = sample["action"].unsqueeze(0).to(device, dtype=dtype)
        
        # 伪造文本 Token (暂时用随机的，代替真正的 tokenizer 输出)
        dummy_text_ids = torch.randint(0, 126000, (batch_size, seq_len), device=device, dtype=torch.long)
        
        logging.info(f"[数据就绪] 真实图片 Shape: {real_images.shape}, 真实动作 Shape: {real_actions.shape}")

    except Exception as e:
        logging.error(f"数据加载失败，请检查 dataset_path 路径是否正确: {e}")
        return

    # ==========================================
    # 4. 真实前向传播 & 计算 Loss
    # ==========================================
    logging.info("--- 开始前向传播与 Loss 计算 ---")
    try:
        # 【核心：冻结 VLM】
        # 我们用 torch.no_grad() 包裹 VLM，这样它的计算图不会保存，既省显存又不更新 VLM 权重
        with torch.no_grad():
            image_embs = model.vlm.embed_image(real_images)
            text_embs = model.vlm.embed_language_tokens(dummy_text_ids)
            prefix_embs = torch.cat([image_embs, text_embs], dim=1) # 拼成 [1, 1369+32, 4096]
            logging.info(f"[成功] VLM 前向完成，联合特征 (Prefix) Shape: {prefix_embs.shape}")

        # 制造假后缀给 Action Expert (模拟 OpenPI 里的 noisy action tokens)
        suffix_length = 32 
        suffix_embs = torch.randn(batch_size, suffix_length, 4096, device=device, dtype=dtype)
        # 告诉 PyTorch 我们要计算 Suffix（也就是 Expert 输入）的梯度
        suffix_embs.requires_grad = True 
        
        # 送入 Action Expert 联合前向
        outputs, _ = model.forward(inputs_embeds=[prefix_embs, suffix_embs])
        expert_out = outputs[1] # 取出专家模型的输出: [1, 32, 4096]
        logging.info(f"[成功] Action Expert 前向完成，输出 Shape: {expert_out.shape}")
        
        # 搞一个简单的线性层，把 4096 维的特征压缩成 7 维的动作
        # (在真实 OpenPI 框架里，这部分是由 pi0_loss.py 里的投影层做的)
        action_head = torch.nn.Linear(4096, 7).to(device, dtype=dtype)
        
        # 拿专家输出的最后一个 token 去预测动作
        pred_action = action_head(expert_out[:, -1, :]) 
        
        # 计算 MSE Loss (预测动作 vs 真实录像动作)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred_action, real_actions)
        
        logging.info(f"🎉 成功算出真实 Loss: {loss.item():.4f}")

    except Exception as e:
        import traceback
        logging.error(f"[失败] 前向传播报错:")
        traceback.print_exc()  
        return

    # ==========================================
    # 5. 反向传播 (Backward) 测试
    # ==========================================
    logging.info("--- 开始反向传播 (只更新 Expert) ---")
    try:
        loss.backward()
        
        # 验证一下：action_head 是我们刚才建的，它应该有梯度
        if action_head.weight.grad is not None:
            logging.info("✅ 梯度检测成功！反向传播没有阻碍，Action Expert 已经具备学习能力！")
        else:
            logging.warning("⚠️ 警告：未检测到梯度，计算图可能断裂。")
            
    except Exception as e:
        import traceback
        logging.error(f"[失败] 反向传播报错:")
        traceback.print_exc()  
        return
        
    logging.info("=== 核心管道测试全部通过！您可以放心去编写 OpenPI 的训练配置了！ ===")

if __name__ == "__main__":
    test_lamda_shapes()