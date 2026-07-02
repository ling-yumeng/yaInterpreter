import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import AutoProcessor, SiglipVisionModel
import os

class yaVisionProjector(nn.Module):
    """
    【多模态投影层 mmproj】
    它的唯一使命：作为视觉空间与文本空间对齐的‘翻译官’。
    将 SigLIP 输出的 768 维视觉特征向量，无损拉伸并转换成 Qwen3.5 脑子能听懂的 2560 维文本 Embedding。
    """
    def __init__(self, vision_dim=768, llm_dim=2560):
        super().__init__()
        # 第一层线性映射：把 768 维暴力拉伸到 2560 维
        self.linear_1 = nn.Linear(vision_dim, llm_dim)
        # GELU 激活函数：引入非线性拟合能力，防止多层线性层退化或塌陷
        self.gelu = nn.GELU()
        # 第二层线性映射：对拉伸后的 2560 维高维特征进行精细映射与稳定
        self.linear_2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        # 典型的 MLP（多层感知机）前向传播数据流
        return self.linear_2(self.gelu(self.linear_1(x)))


class yaVisionAligner(nn.Module):
    """
    【yaVision 多模态异构组装主网络】
    负责统一调度视觉特征提取、投影层变换、文本嵌入拼接以及大模型 Loss 计算。
    """
    def __init__(
        self, 
        # 【修改点】这里可以直接传入本地的文件夹路径字符串
        llm_id="./models/Qwen3.5-4B", 
        vision_id="./models/siglip-base-patch16-384", 
        use_cpu_vision=False
    ):
        super().__init__()
        self.use_cpu_vision = use_cpu_vision
        
        # 显式检查本地路径是否存在，利于 Debug
        if not os.path.exists(vision_id):
            raise FileNotFoundError(f"找不到本地视觉模型目录: {vision_id}")
        if not os.path.exists(llm_id):
            raise FileNotFoundError(f"找不到本地语言模型目录: {llm_id}")
        
        # ----------------------------------------------------------------------
        # 第一步：从本地目录载入 SigLIP
        # ----------------------------------------------------------------------
        print(f"[yaVision] 正在从本地载入视觉发动机: {vision_id}")
        self.vision_processor = AutoProcessor.from_pretrained(vision_id)
        # transformers 会自动在 vision_id 目录下寻找 *.safetensors 文件
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_id)
        
        if self.use_cpu_vision:
            self.vision_encoder = self.vision_encoder.to(torch.float32).cpu()
        else:
            self.vision_encoder = self.vision_encoder.to(torch.bfloat16).cuda()
            
        for param in self.vision_encoder.parameters():
            param.requires_grad = False 
            
        # ----------------------------------------------------------------------
        # 第二步：从本地目录以 4-bit 量化载入 Qwen3.5-4B
        # ----------------------------------------------------------------------
        print(f"[yaVision] 正在从本地量化载入语言模型基座: {llm_id}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        # 同样，直接读取本地的 config.json
        self.llm_config = AutoConfig.from_pretrained(llm_id)
        # 直接读取本地的 safetensors 权重
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            quantization_config=bnb_config,
            device_map="cuda" if not use_cpu_vision else {"": 0}
        )
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # ----------------------------------------------------------------------
        # 第三步：实例化桥梁 mmproj (保持不变)
        # ----------------------------------------------------------------------
        self.mmproj = yaVisionProjector(vision_dim=768, llm_dim=self.llm_config.hidden_size)
        self.mmproj = self.mmproj.to(torch.bfloat16).cuda()

    def forward(self, pixel_values, input_ids, labels=None):
        """
        前向传播：完成视觉与文本在时间轴/空间轴上的完美缝合
        """
        # --- 动作 1：提取图像高级特征 ---
        if self.use_cpu_vision:
            # 异构流：让 CPU 在内存里安安静静把图看完，因为冻结了，所以不需要保存任何中间激活值（no_grad）
            with torch.no_grad():
                img_feats = self.vision_encoder(pixel_values.cpu().float()).last_hidden_state
                # 看完之后，吐出的只是一个极小的矩阵 [1, 576, 768]（几 KB 大小），一枪通过 PCIe 打回 GPU
                img_feats = img_feats.cuda().to(torch.bfloat16)
        else:
            # 极限流：视觉也在 GPU 里跑
            with torch.no_grad():
                img_feats = self.vision_encoder(pixel_values.cuda().to(torch.bfloat16)).last_hidden_state
        
        # --- 动作 2：穿过投影桥梁 ---
        # 张量变形记：[Batch, 576, 768] -> [Batch, 576, 2560]
        visual_embeds = self.mmproj(img_feats)
        
        # --- 动作 3：把用户输入的提示词文本变成高维向量 ---
        # 形状：[Batch, Text_Seq_Len, 2560]
        text_embeds = self.llm.get_input_embeddings()(input_ids.cuda())
        
        # --- 动作 4：史诗级拼接 ---
        # 把 576 个视觉 Token 和 Text_Seq_Len 个文本 Token 连成一条扁平长蛇
        # 形状变化：[Batch, 576 + Text_Seq_Len, 2560]
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # --- 动作 5：送入大模型，计算自回归损失值（Loss） ---
        # 此时大模型看到的东西就像是：‘[一堆伪装成文本的像素] 识别图中的文字：’ 
        outputs = self.llm(inputs_embeds=combined_embeds, labels=labels)
        return outputs.loss