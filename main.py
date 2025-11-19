"""
main.py
狼人杀智能体训练主程序
"""

import os
import torch
import numpy as np
import json
from model import WolfensteinBART
from trainer import PPOTrainer
from environment import WolfensteinEnv
from action_generator import ActionGenerator
from transformers import BertTokenizer, BartModel

def main():
    """主训练函数"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置参数
    config = {
        "bart_model_path": "E:/大模型/bart-base-chinese/",
        "num_episodes": 10000,
        "max_steps_per_episode": 100,
        "lr": 2e-5,  # BART学习率通常稍低
        "gamma": 0.99,
        "clip_epsilon": 0.2,
        "save_interval": 1000,
        "log_interval": 100
    }

    # 直接定义模型路径（替换为你的实际路径）
    bart_model_path = "E:\\大模型\\bart-base-chinese\\"  
    
    # 检查模型文件是否存在
    required_files = [
        os.path.join(bart_model_path, "pytorch_model.bin"),
        os.path.join(bart_model_path, "config.json"),
        os.path.join(bart_model_path, "vocab.txt"),
        os.path.join(bart_model_path, "tokenizer_config.json")
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
    
    # 加载模型和分词器
    print("加载BART模型...")
    model = WolfensteinBART(bart_model_path, device)
    tokenizer = BertTokenizer.from_pretrained(bart_model_path)
    
    # 初始化训练器和环境
    trainer = PPOTrainer(model, tokenizer, lr=config["lr"], gamma=config["gamma"], 
                        clip_epsilon=config["clip_epsilon"])
    env = WolfensteinEnv()
    action_generator = ActionGenerator(model, tokenizer, device)
    
    # 训练统计
    total_rewards = []
    win_rates = {"wolf": [], "villager": []}
    loss_history = []
    
    print("开始训练...")
    for episode in range(config["num_episodes"]):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss = 0
        
        for step in range(config["max_steps_per_episode"]):
            if done:
                break
            
            # 生成动作
            action_idx_logprob, value, action, discussion = action_generator.generate_action(state)
            action_idx = action_idx_logprob[0]    # 动作索引（用于PPO）
            log_prob = action_idx_logprob[1]      # 对数概率
            

            print(f"生成动作：{action}")
            print(f"对数概率：{log_prob}")
            print(f"价值函数值：{value}")
            print(f"讨论内容：{discussion}")
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 收集轨迹
            trainer.collect_trajectory(state, action_idx, log_prob, reward, next_state, done)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
        
        # 更新模型
        loss = trainer.update()
        loss_history.append(loss)
        
        # 记录统计信息
        total_rewards.append(episode_reward)
        
        # 检查胜利情况
        if env.game_state["game_over"]:
            winner = env.game_state["winner"]
            self_role = env.game_state["self_role"]
            if (self_role == "wolf" and winner == "wolf") or (self_role != "wolf" and winner == "villager"):
                win_rates[self_role.split("_")[0]].append(1)
            else:
                win_rates[self_role.split("_")[0]].append(0)
        
        # 打印进度
        if (episode + 1) % config["log_interval"] == 0:
            avg_reward = np.mean(total_rewards[-config["log_interval"]:])
            avg_loss = np.mean(loss_history[-config["log_interval"]:])
            
            # 计算胜率
            wolf_win_rate = np.mean(win_rates["wolf"][-config["log_interval"]:]) if win_rates["wolf"] else 0
            villager_win_rate = np.mean(win_rates["villager"][-config["log_interval"]:]) if win_rates["villager"] else 0
            
            print(f"Episode {episode+1}/{config['num_episodes']}")
            print(f"平均奖励: {avg_reward:.2f}, 平均损失: {avg_loss:.4f}")
            print(f"狼人胜率: {wolf_win_rate:.2%}, 好人胜率: {villager_win_rate:.2%}")
            print("-" * 50)
        
        # 保存模型
        if (episode + 1) % config["save_interval"] == 0:
            model_save_path = f"./models/wolfenstein_bart_ep{episode+1}.bin"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到: {model_save_path}")
    
    # 保存最终模型
    final_model_path = "./models/wolfenstein_bart_final.bin"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 保存训练记录
    training_log = {
        "total_rewards": total_rewards,
        "loss_history": loss_history,
        "wolf_win_rates": win_rates["wolf"],
        "villager_win_rates": win_rates["villager"]
    }
    
    with open("./training_log.json", "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    
    print("训练完成！")
    return model

if __name__ == "__main__":
    # 训练智能体
    trained_model = main()