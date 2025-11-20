class WolfensteinEnv:
    def __init__(self):
        # 初始化游戏核心属性
        self.roles = {}          # 玩家角色：{player_id: "wolf/villager/god"}
        self.alive_players = []  # 存活玩家列表
        self.current_player = 0  # 当前行动玩家
        self.phase = "day"       # 游戏阶段：day/night
        self.round = 1           # 游戏回合数
        self.discussion_history = []  # 讨论记录
        self.voting_history = []      # 投票记录
        self.action_history = []      # 行动记录
        self.done = False        # 游戏是否结束
        self.winner = None       # 胜者：wolf/villager
        
        # 重置环境，初始化状态
        self.reset()

    def reset(self):
        """重置游戏环境，返回初始状态"""
        # 9人局配置：3狼（0,3,7）+ 4村民 + 2神
        self.roles = {
            0: "wolf",    # 智能体固定为狼人（可调整）
            1: "villager",
            2: "villager",
            3: "wolf",
            4: "villager",
            5: "god",
            6: "villager",
            7: "wolf",
            8: "god"
        }
        self.alive_players = list(range(9))  # 初始所有玩家存活
        self.current_player = 0              # 智能体先行动
        self.phase = "day"
        self.round = 1
        self.discussion_history = []
        self.voting_history = []
        self.action_history = []
        self.done = False
        self.winner = None
        
        # 返回初始游戏状态（核心修复：构建game_state字典）
        return self.get_state()

    def get_state(self):
        """构建并返回游戏状态字典（修复game_state缺失问题）"""
        game_state = {
            "roles": self.roles.copy(),
            "alive_players": self.alive_players.copy(),
            "current_player": self.current_player,
            "phase": self.phase,
            "round": self.round,
            "discussion_history": self.discussion_history.copy(),
            "voting_history": self.voting_history.copy(),
            "action_history": self.action_history.copy(),
            "done": self.done,
            "self_role": self.roles[0]  # 智能体的角色（固定为玩家0）
        }
        return game_state

    def _check_winner(self):
        """检查游戏胜负，更新done和winner"""
        # 统计存活狼人/好人数量
        alive_wolves = [p for p in self.alive_players if self.roles[p] == "wolf"]
        alive_good = [p for p in self.alive_players if self.roles[p] != "wolf"]
        
        # 狼人胜利条件：好人数量 ≤ 狼人数量
        if len(alive_good) <= len(alive_wolves):
            self.winner = "wolf"
            self.done = True
        # 好人胜利条件：无存活狼人
        elif len(alive_wolves) == 0:
            self.winner = "villager"
            self.done = True
        # 游戏未结束
        else:
            self.winner = None
            self.done = False

    def _handle_day_action(self, action):
        """处理白天行动（讨论/投票）"""
        reward = 0.05  # 基础发言奖励
        
        # 处理讨论动作
        if action["type"] == "discussion":
            content = action.get("content", "无发言")
            self.discussion_history.append({
                "player": self.current_player,
                "content": content
            })
            # 长发言额外奖励
            if len(content) > 10:
                reward += 0.05
        
        # 处理投票动作
        elif action["type"] == "vote":
            target = action.get("target", 0)
            if target in self.alive_players:
                self.voting_history.append({
                    "voter": self.current_player,
                    "target": target
                })
                # 投票淘汰玩家（简化逻辑：直接淘汰目标）
                self.alive_players.remove(target)
                self.action_history.append(f"玩家{self.current_player}投票淘汰玩家{target}")
                reward += 0.1  # 投票奖励
                # 淘汰后检查胜负
                self._check_winner()
        
        # 切换下一个存活玩家
        if self.current_player in self.alive_players:
            idx = self.alive_players.index(self.current_player)
            next_idx = (idx + 1) % len(self.alive_players)
            self.current_player = self.alive_players[next_idx]
        else:
            self.current_player = self.alive_players[0] if self.alive_players else 0
        
        # 白天结束切换到夜晚
        if len(self.discussion_history) >= len(self.alive_players):
            self.phase = "night"
        
        return self.get_state(), reward, self.done

    def _handle_night_action(self, action):
        """处理夜晚行动（狼人刀人）"""
        reward = 0.0
        if self.roles[self.current_player] == "wolf" and action["type"] == "wolf_kill":
            target = action.get("target", 0)
            if target in self.alive_players and self.roles[target] != "wolf":
                self.alive_players.remove(target)
                self.action_history.append(f"狼人{self.current_player}夜晚刀掉玩家{target}")
                reward += 0.2  # 狼人刀人奖励
                # 刀人后检查胜负
                self._check_winner()
        
        # 夜晚结束切换到白天，回合+1
        self.phase = "day"
        self.round += 1
        
        # 切换下一个玩家
        if self.alive_players:
            self.current_player = self.alive_players[0]
        
        return self.get_state(), reward, self.done

    def step(self, action):
        """执行动作，返回(next_state, reward, done, info)"""
        if self.done:
            return self.get_state(), 0.0, True, {"winner": self.winner}
        
        # 根据阶段处理动作
        if self.phase == "day":
            next_state, reward, done = self._handle_day_action(action)
        elif self.phase == "night":
            next_state, reward, done = self._handle_night_action(action)
        else:
            next_state = self.get_state()
            reward = 0.0
            done = self.done
        
        # 胜负奖励（放大权重，引导策略）
        if self.done:
            if self.winner == "wolf" and self.roles[0] == "wolf":
                reward += 10.0  # 智能体（狼人）胜利奖励
            elif self.winner == "villager" and self.roles[0] != "wolf":
                reward += 10.0  # 智能体（好人）胜利奖励
            else:
                reward -= 5.0   # 失败惩罚
        
        # 返回信息包含胜者
        info = {"winner": self.winner}
        return next_state, reward, done, info