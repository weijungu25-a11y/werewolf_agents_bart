import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BertTokenizer

class WolfensteinBART(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.dtype = torch.float32  # 全FP32，禁用混合精度
        
        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # 加载BART模型（全FP32）
        self.bart = BartForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype
        ).to(self.device)
        
        # 任务头全FP32
        self.hidden_size = self.bart.config.hidden_size
        self.action_head = nn.Linear(self.hidden_size, 6).to(self.device)
        self.value_head = nn.Linear(self.hidden_size, 1).to(self.device)
        self.target_head = nn.Linear(self.hidden_size, 9).to(self.device)
        
    def forward(self, input_ids, attention_mask):
        """前向传播：全FP32，无混合精度"""
        input_ids = input_ids.to(self.device, dtype=torch.long)
        attention_mask = attention_mask.to(self.device, dtype=torch.long)
        
        # 移除autocast上下文，直接FP32计算
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_hidden = encoder_outputs.last_hidden_state[:, 0, :]
        
        action_logits = self.action_head(cls_hidden)
        value = self.value_head(cls_hidden)
        target_logits = self.target_head(cls_hidden)
        
        return action_logits, value, target_logits
    
    def generate_discussion(self, input_ids, attention_mask, max_length=150):
        input_ids = input_ids.to(self.device, dtype=torch.long)
        attention_mask = attention_mask.to(self.device, dtype=torch.long)
        
        # 移除autocast，直接生成
        generated_ids = self.bart.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        return generated_ids