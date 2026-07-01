import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoTokenizer
from model_yavision import yaVisionAligner # 引入你之前写的组装类

# ==============================================================================
# 1. 打造硬核多模态本地数据集载入器 (Dataset)
# ==============================================================================
class yaLocalOCRDataset(Dataset):
    """
    负责从本地读取你的游戏截图和对应的日韩中英歌词/文本标签
    """
    def __init__(self, image_dir, label_file, tokenizer, vision_processor):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.processor = vision_processor
        
        # 假设你的 label_file 是一个文本，每行格式为: "image_name.png\t对应识别的文本答案"
        self.data_samples = []
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    img_name, text = line.strip().split("\t")
                    self.data_samples.append((img_name, text))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        img_name, target_text = self.data_samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 1. 载入本地图片并过 SigLIP 预处理器，强制缩放到 384x384 并归一化
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        # pixel_values 形状: [3, 384, 384]

        # 2. 构造标准的文本输入流
        prompt = "识别图中的所有文字，并给出空间关联:\n"
        answer = target_text + "\n" # 加上结束符让大模型学会收尾
        
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        
        # 拼接成完整的 input_ids 给大模型做自回归上下文
        # 形状: [Prompt_Len + Answer_Len]
        input_ids = torch.tensor(prompt_ids + answer_ids, dtype=torch.long)
        
        # 3. 核心构建：计算掩码 Labels
        # 视觉 Token 占 576 位，Prompt 占 prompt_ids 位，全部用 -100 屏蔽！
        labels = torch.cat([
            torch.full((576,), -100, dtype=torch.long),
            torch.full((len(prompt_ids),), -100, dtype=torch.long),
            torch.tensor(answer_ids, dtype=torch.long) # 只有这里才计算 Loss
        ])
        
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long), # 传给前向拼接的只需要 Prompt 部分
            "labels": labels
        }

# ==============================================================================
# 2. 核心主循环：调用两个类开启炼丹
# ==============================================================================
def run_local_training():
    # 强制开启完全断网离线炼丹模式，防止 transformers 悄悄联网挂起
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 本地路径定义（替换为你真实的本地 safetensors 权重目录）
    LOCAL_LLM_PATH = "./models/Qwen3.5-4B"
    LOCAL_VISION_PATH = "./models/siglip-base-patch16-384"
    
    # 1. 初始化模型
    # 第一次运行默认开启纯 GPU 流(use_cpu_vision=False)。如果爆显存，请手动改为 True 触发 CPU 防御流。
    model = yaVisionAligner(
        llm_id=LOCAL_LLM_PATH, 
        vision_id=LOCAL_VISION_PATH, 
        use_cpu_vision=False 
    )
    
    # 2. 初始化分词器和视觉处理器
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_PATH)
    
    # 3. 准备你本地的数据集（这里假设你已经创建了对应的文件夹和描述文本）
    # mock 一个本地的 labels.txt。真实开发时请准备好对应的图片和文本
    if not os.path.exists("./train_data"):
        print("⚠️ 未检测到本地训练数据目录，正在自动创建样本占位符...")
        os.makedirs("./train_data/images", exist_ok=True)
        with open("./train_data/labels.txt", "w") as f:
            f.write("sample_01.png\t漂泊ノ海 -Wandering Sea-\n")
        # 随手画一张假图用于跑通测试
        Image.fromarray(axis=0, obj=None).resize((384,384)).save("./train_data/images/sample_01.png")

    dataset = yaLocalOCRDataset(
        image_dir="./train_data/images",
        label_file="./train_data/labels.txt",
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
        torch.save(model.mmproj.state_dict(), os.path.join(save_dir, "mmproj.bin"))
        print(f"💾 已成功将当前 Epoch 的抽象变换连接件保存至: {save_dir}/mmproj.bin")

if __name__ == "__main__":
    run_local_training()