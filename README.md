# werewolf_agents_bart
狼人杀智能体训练框架
基于 BART-base-chinese 和 PPO 算法的狼人杀智能体训练框架
项目结构

wolfenstein_agent/
├── model.py           # BART模型定义
├── trainer.py         # PPO训练器
├── environment.py     # 游戏环境模拟器
├── action_generator.py # 动作生成器
├── main.py            # 主训练程序
├── utils.py           # 工具函数
├── README.md          # 使用说明
└── requirements.txt   # 依赖包列表
功能特点
1. 基于 BART 的中文理解与生成
•使用 bart-base-chinese 模型，专门针对中文优化
•Encoder-Decoder 结构，同时支持理解和生成任务
•生成自然流畅的中文讨论内容
2. PPO 强化学习算法
•近端策略优化算法，训练稳定
•支持轨迹收集和批量更新
•自适应学习率和梯度裁剪
3. 完整的狼人杀游戏环境
•9 人标准狼人杀规则
•5 种角色：村民、狼人、预言家、女巫、猎人
•夜晚 / 白天 / 投票三个游戏阶段
4. 智能动作生成
•根据角色和游戏阶段生成合理动作
•支持杀人、查验、用药、投票等操作
•角色专属的讨论策略
安装依赖

pip install -r requirements.txt
requirements.txt 内容：

torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
准备模型文件
需要下载 bart-base-chinese 模型文件到项目目录：

bart-base-chinese/
├── config.json
├── pytorch_model.bin
├── vocab
├── tokenizer_config.json
└── special_tokens_map.json
启动训练

python main.py
训练参数配置
在 main.py 中可以调整以下参数：

config = {
    "bart_model_path": "./bart-base-chinese",  # 模型路径
    "num_episodes": 10000,                    # 训练轮数
    "max_steps_per_episode": 100,             # 每轮最大步数
    "lr": 2e-5,                               # 学习率
    "gamma": 0.99,                            # 折扣因子
    "clip_epsilon": 0.2,                      # PPO剪辑系数
    "save_interval": 1000,                    # 模型保存间隔
    "log_interval": 100                       # 日志输出间隔
}
训练过程
训练过程中会输出以下信息：
•每 100 轮的平均奖励和损失
•狼人和好人阵营的胜率
•模型保存进度
输出文件
训练完成后会生成：
•models/目录：保存训练过程中的模型权重
•training_log.json：训练记录，包含奖励、损失、胜率等数据
模型性能评估
训练日志包含以下指标：
•平均奖励：反映智能体的整体表现
•损失值：PPO 损失和价值损失
•胜率：狼人和好人阵营的获胜概率
扩展功能
1. 添加更多角色
可以在 environment.py 中扩展角色系统，如守卫、白痴等。
2. 优化讨论生成
可以添加更多角色专属的讨论模板，或使用更复杂的生成策略。
3. 多智能体训练
可以扩展为多智能体训练，让多个 AI 相互对抗。
4. 人类对战模式
可以添加人类玩家接口，实现 AI 与人类的对战。
技术优势
1.中文优化：使用专门的中文预训练模型
2.生成能力强：BART 的 Encoder-Decoder 结构适合对话生成
3.训练稳定：PPO 算法确保训练过程稳定收敛
4.游戏逻辑完整：实现了标准的狼人杀游戏规则
5.易于扩展：模块化设计，便于添加新功能
注意事项
1.训练需要 GPU 支持，建议使用 RTX 5080 或更高配置
2.完整训练 10000 轮大约需要 2-3 天时间
3.可以根据需要调整训练轮数和批次大小
4.模型文件较大（约 500MB），需要确保磁盘空间充足
许可证
本项目采用 MIT 许可证，欢迎使用和修改。
（注：文档部分内容可能由 AI 生成）