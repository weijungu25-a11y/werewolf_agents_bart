import torch
from model import WolfensteinBART  # 导入你定义的模型类
from environment import WolfensteinEnv  # 导入狼人杀环境
from transformers import BertTokenizer  # 导入分词器
from action_generator import ActionGenerator  # 导入动作生成器

# ===================== 1. 基础配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model_path = "E:/大模型/bart-base-chinese/"  # 原BART预训练模型路径
saved_model_path = "./models/wolfenstein_bart_final.bin"  # 保存的智能体模型路径

# ===================== 2. 加载模型+分词器 =====================
# 初始化空模型（结构和训练时一致）
model = WolfensteinBART(bart_model_path, device)
# 加载保存的权重
checkpoint = torch.load(saved_model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 切换到推理模式（禁用Dropout，固定权重）
print("模型加载完成！")

# 加载分词器（和训练时一致）
tokenizer = BertTokenizer.from_pretrained(bart_model_path)

# ===================== 3. 初始化环境+动作生成器 =====================
env = WolfensteinEnv()  # 初始化狼人杀环境
action_generator = ActionGenerator(model, tokenizer, device)  # 动作生成器（连接模型和环境）

# ===================== 4. 智能体推理+游戏交互 =====================
# 重置环境，获取初始游戏状态
state = env.reset()
done = False
total_reward = 0.0

print("===== 开始狼人杀游戏（智能体自主决策） =====")
while not done:
    # 关键：模型生成动作（无需训练，纯推理）
    with torch.no_grad():  # 禁用梯度计算，节省显存+提速
        action_idx_logprob, value, action, discussion = action_generator.generate_action(state)
    
    # 打印智能体决策（直观看到动作）
    print(f"\n当前游戏阶段：{state['phase']} | 智能体角色：{state['self_role']}")
    print(f"智能体动作类型：{action['type']}")
    if action['type'] == "discussion":
        print(f"智能体发言：{discussion[:50]}...")  # 只打印前50字
    elif action['type'] in ["vote", "wolf_kill"]:
        print(f"智能体目标玩家：{action['target']}")
    
    # 执行动作，更新环境
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    
    # 更新状态，进入下一轮
    state = next_state

# 游戏结束，打印结果
print("\n===== 游戏结束 =====")
print(f"最终胜者：{info['winner']}")
print(f"智能体总奖励：{total_reward:.2f}")
print(f"存活玩家：{state['alive_players']}")