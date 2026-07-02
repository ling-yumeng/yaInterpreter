import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from yaVision import yaVisionAligner  # 引入你之前写的组装类

# ==============================================================================
# 1. 打造硬核多模态本地数据集载入器 (Dataset)
# ==============================================================================
class yaLocalOCRDataset(Dataset):
    """
    负责直接读取 TRDG 原生生成的 labels.txt（格式：文件名 文本答案）
    并加载对应的 <number>.jpg 图片进行多模态特训
    """
    def __init__(self, image_dir, label_file, tokenizer, vision_processor):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = vision_processor
        self.data_samples = []
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"❌ 未在指定路径找到标签文件: {label_file}")
            
        # 直接读取 TRDG 吐出来的 labels.txt
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 核心解析：TRDG 默认使用空格分隔 "<filename> <Texts>"
                # 使用 maxsplit=1 限制切分次数，防止日文文本内部含空格时导致解析错位
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    img_name, target_text = parts[0], parts[1]
                    # 确保图片在本地真实存在，防止空跑触发文件不存在报警
                    if os.path.exists(os.path.join(self.image_dir, img_name)):
                        self.data_samples.append((img_name, target_text))
                        
        print(f"📖 成功载入本地特训数据集，共检测到 {len(self.data_samples)} 个有效样本。")

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        img_name, target_text = self.data_samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 1. 载入本地 .jpg 图片并过 SigLIP 预处理器，强制缩放到 384x384 并归一化
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        # pixel_values 形状: [3, 384, 384]

        # 2. 构造标准的文本输入流
        prompt = "识别图中的所有文字，并给出空间关联:\n"
        answer = target_text + "\n"  # 加上结束符让大模型学会收尾
        
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        
        # 3. 核心构建：计算掩码 Labels
        # 视觉 Token 占 576 位，Prompt 占 prompt_ids 位，全部用 -100 屏蔽！
        # 只有最后真正的日文答案部分才计算真实 CrossEntropyLoss
        labels = torch.cat([
            torch.full((576,), -100, dtype=torch.long),
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            torch.tensor(answer_ids, dtype=torch.long)
        ])
        
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),  # 传给前向拼接的只需要 Prompt 部分
            "labels": labels
        }

# ==============================================================================
# 2. 核心主循环：调用两个类开启炼丹
# ==============================================================================
def run_local_training():
    # 强制开启完全断网离线炼丹模式，防止 transformers 悄悄联网挂起
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 本地模型与生成的特训数据集路径定义
    LOCAL_LLM_PATH = "./models/Qwen3.5-4B"
    LOCAL_VISION_PATH = "./models/siglip2-base-patch16-384"
    DATASET_DIR = "./ja_corrupted_images"
    LABEL_FILE_PATH = "./ja_corrupted_images/labels.txt"
    
    # 1. 初始化多模态组装模型
    # 第一次运行默认开启纯 GPU 流(use_cpu_vision=False)。如果爆显存，请手动改为 True 触发 CPU 防御流。
    model = yaVisionAligner(
        llm_id=LOCAL_LLM_PATH, 
        vision_id=LOCAL_VISION_PATH, 
        use_cpu_vision=False 
    )
    
    # 2. 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_PATH)
    
    # 3. 准备你本地刚生成的 TRDG 数据集
    dataset = yaLocalOCRDataset(
        image_dir=DATASET_DIR,
        label_file=LABEL_FILE_PATH,
        tokenizer=tokenizer,
        vision_processor=model.vision_processor
    )
    
    # 4. 用 DataLoader 包裹，BatchSize=1（极限压榨 8GB 显存配置）
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 5. 绑定优化器：死死锁定，只更新 mmproj 参数！
    optimizer = torch.optim.AdamW(model.mmproj.parameters(), lr=1e-4, weight_decay=0.1)
    
    model.train()
    print("\n🚀 [yaInterpreter] 离线多模态空间对齐特训正式开启！")
    
    # 开启 Epoch 循环
    for epoch in range(10):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 把 DataLoader 吐出来的数据统一打上显卡
            pixel_values = batch["pixel_values"].cuda()
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            
            # 前向传播：缝合空间特征，激活 Qwen3.5 词表 Softmax
            loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            
            # 反向传播：梯度穿过冻结层，汇聚至 mmproj
            loss.backward()
            
            # 梯度裁剪：防止在训练畸变字体初期，梯度过大导致 mmproj 的权重直接炸成 NaN
            torch.nn.utils.clip_grad_norm_(model.mmproj.parameters(), max_norm=1.0)
            
            # 更新 mmproj 的权重参数
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Epoch [{epoch+1}/10] | Step {step+1} | 当前单步交叉熵 Loss: {loss.item():.4f}")
            
        # 每个 Epoch 结束后，立刻保存阶段性的 mmproj 本地权重
        save_dir = f"./checkpoints/epoch_{epoch+1}"
        os.makedirs(save_dir, exist_ok=True)
        # 只保存 mmproj 即可，千万别把几 GB 冻结的 Qwen3.5 重新存一遍！
        torch.save(model.state_dict(), os.path.join(save_dir, "mmproj.bin"))
        print(f"💾 已成功将当前 Epoch 的抽象变换连接件保存至: {save_dir}/mmproj.bin")

if __name__ == "__main__":
    run_local_training()