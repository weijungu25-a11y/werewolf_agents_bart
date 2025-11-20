import torch
import torch.nn.functional as F
import random
from transformers import BertTokenizer

class ActionGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def encode_state(self, state):
        """编码游戏状态为模型输入（兼容缺失键）"""
        # 安全获取值，缺失则用默认值
        self_role = state.get("self_role", "villager")
        alive_players = state.get("alive_players", [0,1,2,3,4,5,6,7,8])
        phase = state.get("phase", "day")
        history = state.get("history", "")
        
        # 拼接游戏状态文本
        state_text = f"""
        角色：{self_role}
        存活玩家：{alive_players}
        游戏阶段：{phase}
        历史行动：{history}
        """
        # 编码为张量并移到设备
        inputs = self.tokenizer(
            state_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs["input_ids"] = inputs["input_ids"].to(self.device)
        inputs["attention_mask"] = inputs["attention_mask"].to(self.device)
        return inputs
    
    # 新增：动作类型映射
    ACTION_TYPES = {
        0: "wolf_kill",          # 狼人杀人
        1: "seer_divination",    # 预言家查验
        2: "witch_heal",         # 女巫治疗
        3: "witch_poison",       # 女巫下毒
        4: "discussion",         # 讨论发言
        5: "vote"                # 投票
    }
    #添加反向映射
    ACTION_TYPE_TO_INDEX = {v: k for k, v in ACTION_TYPES.items()}
    
    def generate_action(self, state):
      """生成动作（修复元组索引+KeyError）"""
    # 编码游戏状态
      encoded_state = self.encode_state(state)
      input_ids = encoded_state["input_ids"]
      attention_mask = encoded_state["attention_mask"]
    
    # 模型前向传播（返回元组：(action_logits, value, target_logits)）
      with torch.no_grad():
          outputs = self.model(input_ids, attention_mask)
    
    # 元组索引（0:动作logits, 1:价值, 2:目标logits）
      action_logits = outputs[0]
      value = outputs[1]
      target_logits = outputs[2]
    
    # 计算动作概率和log概率
      action_probs = F.softmax(action_logits, dim=-1)
      action_dist = torch.distributions.Categorical(action_probs)
      action_idx = action_dist.sample()
      log_prob = action_dist.log_prob(action_idx)  # 保持为 tensor
    
    # 生成目标玩家
      target_probs = F.softmax(target_logits, dim=-1)
      target_dist = torch.distributions.Categorical(target_probs)
      target = target_dist.sample()
    
    # 生成讨论内容
      discussion = self._generate_discussion(state, encoded_state)
    
    # 修复：创建动作字典而不是返回整数
      action = {
          "type": self.ACTION_TYPES.get(action_idx.item(), "vote"),
          "target": target.item(),
          "content": discussion if self.ACTION_TYPES.get(action_idx.item()) == "discussion" else ""
    }
    
    # 返回结构化结果 - 不要将 log_prob 转换为 float
      return (action_idx, log_prob), value.item(), action, discussion
    
    def _generate_discussion(self, state, encoded_state):
        """生成讨论内容（修复设备+缺失键）"""
        # 安全获取角色
        role = state.get("self_role", "villager")
        
        role_prompts = {
            "villager": "作为村民，我需要：\n1. 表明自己的好人身份\n2. 分析其他玩家的可疑行为\n3. 提出合理的投票建议\n4. 保持逻辑清晰的表达\n\n我的发言：",
            "wolf": "作为狼人，我需要：\n1. 伪装成好人身份\n2. 误导其他玩家怀疑好人\n3. 避免暴露自己的狼人身份\n4. 提出有利于狼人的投票建议\n\n我的发言：",
            "seer": "作为预言家，我需要：\n1. 表明自己的预言家身份\n2. 分享查验信息（真假结合）\n3. 引导好人投票狼人\n4. 建立自己的可信度\n\n我的发言：",
            "witch": "作为女巫，我需要：\n1. 谨慎使用毒药和解药\n2. 根据夜晚信息分析局势\n3. 在合适时机表明身份\n4. 保护关键好人角色\n\n我的发言：",
            "hunter": "作为猎人，我需要：\n1. 表明自己的猎人身份\n2. 威胁狼人不敢轻易投我\n3. 分析其他玩家的身份\n4. 在被投死时带走可疑玩家\n\n我的发言："
        }
        
        role_prompt = role_prompts.get(role, role_prompts["villager"])
        prompt_text = f"{role_prompt}\n"
        
        # 编码提示并移到设备
        prompt_inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        prompt_input_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_attention_mask = prompt_inputs["attention_mask"].to(self.device)
        
        # 编码状态并移到设备
        encoded_input_ids = encoded_state["input_ids"].to(self.device)
        encoded_attention_mask = encoded_state["attention_mask"].to(self.device)
        
        # 拼接张量（确保设备一致）
        combined_input_ids = torch.cat([
            encoded_input_ids,
            prompt_input_ids
        ], dim=1)[:, :512]
        
        combined_attention_mask = torch.cat([
            encoded_attention_mask,
            prompt_attention_mask
        ], dim=1)[:, :512]
        
        # 生成讨论内容
        with torch.no_grad():
            generated_ids = self.model.generate_discussion(
                input_ids=combined_input_ids,
                attention_mask=combined_attention_mask,
                max_length=150
            )
        
        # 解码文本
        discussion_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 提取发言部分
        if "我的发言：" in discussion_text:
            discussion_text = discussion_text.split("我的发言：")[-1].strip()
        
        # 兜底模板（兼容缺失的target）
        if not discussion_text or len(discussion_text) < 5:
            default_templates = {
                "villager": ["我是好人，请大家相信我。", "我觉得玩家{target}行为可疑。"],
                "wolf": ["我是好人，大家不要怀疑我。", "我觉得玩家{target}是狼人。"],
                "seer": ["我是预言家，昨晚查验了玩家{target}，他是好人。", "请大家相信我，我是真预言家。"],
                "witch": ["我是女巫，我有解药和毒药。", "昨晚我救了玩家{target}。"],
                "hunter": ["我是猎人，如果我被投死我会开枪带走一个。", "我觉得玩家{target}有问题。"]
            }
            
            template = random.choice(default_templates.get(role, default_templates["villager"]))
            
            # 安全填充target
            if "{target}" in template:
                alive_others = state.get("alive_players", [0,1,2,3,4,5,6,7,8])
                alive_others = [p for p in alive_others if p != 0]
                if alive_others:
                    target = random.choice(alive_others)
                    template = template.format(target=target)
                else:
                    template = template.replace("{target}", "3")  # 默认玩家3
            
            discussion_text = template
        
        return discussion_text