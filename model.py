import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BertTokenizer

class WolfensteinBART(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 加载带生成头的BART模型（移到GPU）
        self.bart = BartForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(self.device)
        
        # 狼人杀任务头（初始化后立即移到GPU！）
        self.hidden_size = self.bart.config.hidden_size
        self.action_head = nn.Linear(self.hidden_size, 6).to(self.device)   # 移到GPU
        self.value_head = nn.Linear(self.hidden_size, 1).to(self.device)    # 移到GPU
        self.target_head = nn.Linear(self.hidden_size, 9).to(self.device)   # 移到GPU
        
    def forward(self, input_ids, attention_mask):
        """前向传播（确保所有输入在GPU）"""
        # 强制输入张量移到GPU（双重保险）
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # 获取BART编码器输出
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_hidden = encoder_outputs.last_hidden_state[:, 0, :]  # 此时在GPU
        
        # 线性层计算（所有层都在GPU，匹配输入）
        action_logits = self.action_head(cls_hidden)
        value = self.value_head(cls_hidden)
        target_logits = self.target_head(cls_hidden)
        
        return action_logits, value, target_logits
    
    def generate_discussion(self, input_ids, attention_mask, max_length=150):
        # 强制输入移到GPU
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        generated_ids = self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,  # 添加这一行
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        return generated_ids