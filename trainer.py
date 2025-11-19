"""
trainer.py
PPO强化学习训练器
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BartTokenizer
from action_generator import ActionGenerator

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, model, tokenizer, lr=3e-5, gamma=0.99, clip_epsilon=0.2):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = []
        self.device = next(model.parameters()).device
    
    def collect_trajectory(self, state, action, log_prob, reward, next_state, done):
        """收集轨迹数据"""
        self.memory.append((state, action, log_prob, reward, next_state, done))
    
    def compute_returns(self):
        """计算回报值"""
        returns = []
        discounted_return = 0
        
        # 逆序计算回报
        for state, action, log_prob, reward, next_state, done in reversed(self.memory):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        # 标准化回报
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update(self, epochs=10, batch_size=32):
       """PPO更新"""
       if not self.memory:
        return 0.0
    
       states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)
    
       returns = self.compute_returns()
       old_log_probs = torch.stack(old_log_probs).to(self.device)
    
       total_loss = 0.0
    
       for _ in range(epochs):
        # 打乱数据
           indices = torch.randperm(len(states), device=self.device)
        
           for start in range(0, len(states), batch_size):
               end = start + batch_size
               batch_indices = indices[start:end]
            
            # 获取批次数据
               batch_states = [states[i] for i in batch_indices.cpu().numpy()]
               batch_actions = [actions[i] for i in batch_indices.cpu().numpy()]  # 这些是动作索引（整数）
               batch_returns = returns[batch_indices]
               batch_old_log_probs = old_log_probs[batch_indices]
            
            # 编码状态
               batch_inputs = self.encode_states(batch_states)
            
            # 前向传播
               outputs = self.model(
                   input_ids=batch_inputs["input_ids"],
                   attention_mask=batch_inputs["attention_mask"]
               )
               action_logits = outputs[0]
               values = outputs[1].squeeze()
            
            # 计算动作概率
               action_probs = torch.softmax(action_logits, dim=1)
               log_probs = torch.log(action_probs)
            
            # 将动作索引转换为 tensor
               batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
            
            # 选择对应动作的 log_prob
               selected_log_probs = log_probs.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)
            
            # 计算优势
               advantages = batch_returns - values.detach()
            
            # PPO损失
               ratio = torch.exp(selected_log_probs - batch_old_log_probs)
               surr1 = ratio * advantages
               surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
               policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
               value_loss = F.mse_loss(values, batch_returns)
            
            # 总损失
               loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
               self.optimizer.zero_grad()
               loss.backward()
               torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
               self.optimizer.step()
            
               total_loss += loss.item()
       
    # 清空内存
       self.memory.clear()
    
       return total_loss / (epochs * (len(states) // batch_size + 1))
    
    def encode_states(self, states):
        """编码多个状态"""
        encoded_states = []
        
        for state in states:
            encoded_state = self.encode_state(state)
            encoded_states.append(encoded_state)
        
        # 批次处理
        batch_inputs = {
            "input_ids": torch.cat([s["input_ids"] for s in encoded_states]),
            "attention_mask": torch.cat([s["attention_mask"] for s in encoded_states])
        }
        
        return batch_inputs
    
    def encode_state(self, state):
        """将游戏状态编码为BART输入"""
        
        # 基础游戏信息
        game_info = f"游戏阶段：{state['phase']}，当前玩家：{state['current_player']}，" \
                   f"存活玩家：{state['alive_players']}，游戏回合：{state['round']}"
        
        # 角色信息（只有自己知道）
        role_info = f"我的角色：{state['self_role']}"
        
        # 讨论历史
        discussion_text = "讨论历史：\n"
        for msg in state['discussion_history'][-5:]:  # 最近5轮讨论
            discussion_text += f"玩家{msg['player']}：{msg['content']}\n"
        
        # 投票历史
        vote_text = "投票历史："
        for round_votes in state['voting_history'][-3:]:  # 最近3轮投票
            vote_text += f"{round_votes} "
        
        # 游戏行动历史
        action_text = "行动历史："
        for action in state['action_history'][-3:]:  # 最近3轮行动
            action_text += f"{action} "
        
        # 完整输入文本
        input_text = f"{game_info}\n{role_info}\n\n{discussion_text}\n{vote_text}\n{action_text}"
        
        # BART编码
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs