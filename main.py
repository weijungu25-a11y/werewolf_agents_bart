import os
import torch
import numpy as np
import json
# 注释/删除混合精度导入
# from torch.cuda.amp import autocast, GradScaler
from model import WolfensteinBART
from trainer import PPOTrainer
from environment import WolfensteinEnv
from action_generator import ActionGenerator
from transformers import BertTokenizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 简化配置（进一步降低复杂度，快速验证）
    config = {
        "bart_model_path": "E:/大模型/bart-base-chinese/",
        "num_episodes": 10000,  # 先跑10轮，而非100轮，快速验证
        "max_steps_per_episode": 40,  # 每轮仅10步，减少计算
        "lr": 2e-5,
        "gamma": 0.99,
        "clip_epsilon": 0.2,
        "save_interval": 1000,
        "log_interval": 1,  # 每轮打印日志，看是否卡住
        "batch_size": 2,    # 进一步减小批次，降低GPU压力
        "epochs": 1,        # PPO仅1轮更新，减少计算
        "grad_accumulation_steps": 1  # 关闭梯度累积，简化逻辑
    }

    # 检查模型文件（保留）
    bart_model_path = "E:\\大模型\\bart-base-chinese\\"  
    required_files = [
        os.path.join(bart_model_path, "pytorch_model.bin"),
        os.path.join(bart_model_path, "config.json"),
        os.path.join(bart_model_path, "vocab.txt"),
        os.path.join(bart_model_path, "tokenizer_config.json")
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
    
    # 加载模型（禁用混合精度，全FP32）
    print("加载BART模型...")
    model = WolfensteinBART(bart_model_path, device)
    tokenizer = BertTokenizer.from_pretrained(bart_model_path)
    
    # 初始化组件（scaler设为None，禁用梯度缩放）
    trainer = PPOTrainer(model, tokenizer, 
                        lr=config["lr"], 
                        gamma=config["gamma"], 
                        clip_epsilon=config["clip_epsilon"],
                        batch_size=config["batch_size"],
                        epochs=config["epochs"],
                        grad_accumulation_steps=config["grad_accumulation_steps"])
    env = WolfensteinEnv()
    action_generator = ActionGenerator(model, tokenizer, device)
    
    # 注释/删除混合精度初始化
    # scaler = GradScaler() if torch.cuda.is_available() and device.type == "cuda" else None
    
    # 训练统计
    total_rewards = []
    win_rates = {"wolf": [], "villager": []}
    loss_history = []
    
    print("开始训练...")
    for episode in range(config["num_episodes"]):
        # 每轮打印进度，定位是否卡在某轮
        print(f"===== 开始第 {episode+1}/{config['num_episodes']} 轮训练 =====")
        state = env.reset()
        done = False
        episode_reward = 0
        
        for step in range(config["max_steps_per_episode"]):
            # 每步打印，定位是否卡在某步
            print(f"第 {step+1}/{config['max_steps_per_episode']} 步")
            if done:
                break
            
            # 生成动作
            print("生成动作中...")
            action_idx_logprob, value, action, discussion = action_generator.generate_action(state)
            action_idx = action_idx_logprob[0]    
            log_prob = action_idx_logprob[1]  

            # 执行动作
            print("执行动作中...")
            next_state, reward, done, _ = env.step(action)
            # 训练循环内，执行动作后：
            next_state, reward, done, info = env.step(action)  # info包含winner
            winner = info.get("winner")
            
            # 收集轨迹
            print("收集轨迹中...")
            trainer.collect_trajectory(state, action_idx, log_prob, reward, next_state, done)
            # 每轮episode结束后，统计胜率
            if done:
                if winner == "wolf":
                    win_rates["wolf"].append(1.0)
                    win_rates["villager"].append(0.0)
                elif winner == "villager":
                    win_rates["wolf"].append(0.0)
                    win_rates["villager"].append(1.0)
                else:
        # 未分胜负，暂不计入
                    win_rates["wolf"].append(0.0)
                    win_rates["villager"].append(0.0)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
        
        # 移除混合精度上下文，直接更新模型
        print("更新模型中...")
        loss = trainer.update(scaler=None)  # scaler传None，禁用梯度缩放
        loss_history.append(loss)
        
        # 强制释放显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 每轮打印结果
        total_rewards.append(episode_reward)
        print(f"第 {episode+1} 轮结束 | 奖励: {episode_reward:.2f} | 损失: {loss:.4f}")
        
        # 打印进度（log_interval=1，每轮都打印）
        if (episode + 1) % config["log_interval"] == 0:
            avg_reward = np.mean(total_rewards[-config["log_interval"]:]) if total_rewards else 0
            avg_loss = np.mean(loss_history[-config["log_interval"]:]) if loss_history else 0
            
            wolf_win_rate = np.mean(win_rates["wolf"][-config["log_interval"]:]) if win_rates["wolf"] else 0
            villager_win_rate = np.mean(win_rates["villager"][-config["log_interval"]:]) if win_rates["villager"] else 0
            
            print(f"Episode {episode+1}/{config['num_episodes']}")
            print(f"平均奖励: {avg_reward:.2f}, 平均损失: {avg_loss:.4f}")
            print(f"狼人胜率: {wolf_win_rate:.2%}, 好人胜率: {villager_win_rate:.2%}")
            print("-" * 50)
    
    # 保存模型/日志（保留）
    print("训练完成，保存模型...")
    final_model_path = "./models/wolfenstein_bart_final.bin"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'dtype': str(torch.float32)
    }, final_model_path)
    print(f"模型已保存到: {final_model_path}")
    
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
    trained_model = main()