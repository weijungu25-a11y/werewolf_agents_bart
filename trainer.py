import torch
import torch.nn.functional as F
import numpy as np
from transformers import BartTokenizer
from action_generator import ActionGenerator

class PPOTrainer:
    def __init__(self, model, tokenizer, lr=3e-5, gamma=0.99, clip_epsilon=0.2,
                 batch_size=2, epochs=1, grad_accumulation_steps=1):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = []
        self.device = next(model.parameters()).device
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_accumulation_steps = grad_accumulation_steps
    
    def collect_trajectory(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]
    
    def compute_returns(self):
        returns = []
        discounted_return = 0
        for state, action, log_prob, reward, next_state, done in reversed(self.memory):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update(self, scaler=None):  # scaler仅占位，不使用
        if not self.memory:
            return 0.0
        
        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)
        returns = self.compute_returns()
        old_log_probs = torch.stack(old_log_probs).to(self.device, dtype=torch.float32)
        
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        for _ in range(self.epochs):
            indices = torch.randperm(len(states), device=self.device)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = [states[i] for i in batch_indices.cpu().numpy()]
                batch_actions = [actions[i] for i in batch_indices.cpu().numpy()]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                batch_inputs = self.encode_states(batch_states)
                outputs = self.model(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"]
                )
                action_logits = outputs[0]
                values = outputs[1].squeeze()
                
                action_probs = torch.softmax(action_logits, dim=1)
                log_probs = torch.log(action_probs)
                
                batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
                selected_log_probs = log_probs.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1)
                
                advantages = batch_returns - values.detach()
                ratio = torch.exp(selected_log_probs - batch_old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                loss = policy_loss + 0.5 * value_loss
                
                # 移除梯度累积/缩放，直接反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
        
        self.memory.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return total_loss / (self.epochs * (len(states) // self.batch_size + 1))
    
    def encode_states(self, states):
        encoded_states = []
        for state in states:
            encoded_state = self.encode_state(state)
            encoded_states.append(encoded_state)
        batch_inputs = {
            "input_ids": torch.cat([s["input_ids"] for s in encoded_states]).to(self.device),
            "attention_mask": torch.cat([s["attention_mask"] for s in encoded_states]).to(self.device)
        }
        return batch_inputs
    
    def encode_state(self, state):
        game_info = f"游戏阶段：{state['phase']}，当前玩家：{state['current_player']}，" \
                   f"存活玩家：{state['alive_players']}，游戏回合：{state['round']}"
        role_info = f"我的角色：{state['self_role']}"
        discussion_text = "讨论历史：\n"
        for msg in state['discussion_history'][-5:]:
            discussion_text += f"玩家{msg['player']}：{msg['content']}\n"
        vote_text = "投票历史："
        for round_votes in state['voting_history'][-3:]:
            vote_text += f"{round_votes} "
        action_text = "行动历史："
        for action in state['action_history'][-3:]:
            action_text += f"{action} "
        
        input_text = f"{game_info}\n{role_info}\n\n{discussion_text}\n{vote_text}\n{action_text}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        inputs = {
            "input_ids": inputs["input_ids"].to(self.device, dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(self.device, dtype=torch.long)
        }
        return inputs