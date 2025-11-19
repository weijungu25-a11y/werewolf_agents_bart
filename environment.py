"""
environment.py
狼人杀游戏环境模拟器
"""

import random

class WolfensteinEnv:
    """狼人杀游戏环境模拟器"""
    
    def __init__(self, num_players=9):
        self.num_players = num_players
        self.roles = ["villager", "villager", "villager", "wolf", "wolf", "wolf", 
                     "seer", "witch", "hunter"]
        self.reset()
    
    def reset(self):
        """重置游戏环境"""
        # 随机分配角色
        random.shuffle(self.roles)
        
        self.game_state = {
            "phase": "night",  # night, day, vote
            "round": 1,
            "current_player": 0,
            "self_role": self.roles[0],  # 智能体总是玩家0
            "alive_players": list(range(self.num_players)),
            "discussion_history": [],
            "voting_history": [],
            "action_history": [],
            "witch_potion": {"heal": True, "poison": True},
            "game_over": False,
            "winner": None,
            # 新增：history键（核心修复，解决KeyError）
            "history": ""  # 整合所有历史信息的文本，供模型编码使用
        }
        
        return self.get_state()
    
    def get_state(self):
        """获取当前状态（智能体视角）"""
        # 每次获取状态时，同步更新history字段（整合所有历史信息）
        state = self.game_state.copy()
        
        # 拼接历史信息为可读文本（方便模型理解）
        history_parts = []
        
        # 动作历史
        if self.game_state["action_history"]:
            history_parts.append(f"动作记录：{'; '.join(self.game_state['action_history'])}")
        
        # 讨论历史（取最近5条，避免文本过长）
        if self.game_state["discussion_history"]:
            recent_discussions = self.game_state["discussion_history"][-5:]
            discussion_text = []
            for d in recent_discussions:
                discussion_text.append(f"玩家{d['player']}说：{d['content']}")
            history_parts.append(f"讨论记录：{'; '.join(discussion_text)}")
        
        # 投票历史
        if self.game_state["voting_history"]:
            history_parts.append(f"投票记录：{'; '.join(self.game_state['voting_history'])}")
        
        # 更新history字段（无历史则为空字符串）
        state["history"] = " | ".join(history_parts) if history_parts else ""
        
        return state
    
    def step(self, action):
        """执行动作"""
        phase = self.game_state["phase"]
        current_player = self.game_state["current_player"]
        
        # 根据游戏阶段处理动作
        if phase == "night":
            next_state, reward, done = self._handle_night_action(action)
        elif phase == "day":
            next_state, reward, done = self._handle_day_action(action)
        elif phase == "vote":
            next_state, reward, done = self._handle_vote_action(action)
        else:
            next_state, reward, done = self.game_state, 0, False
        
        # 检查游戏是否结束
        if not done:
            self._check_game_over()
        
        return next_state, reward, done, {}
    
    def _handle_night_action(self, action):
        """处理夜晚行动"""
        current_player = self.game_state["current_player"]
        role = self.roles[current_player]
        reward = 0.1  # 基础奖励
        
        # 根据角色处理动作
        if role == "wolf" and action["type"] == "wolf_kill":
            target = action["target"]
            if target in self.game_state["alive_players"]:
                self.game_state["action_history"].append(f"狼杀玩家{target}")
                self.game_state["alive_players"].remove(target)
                reward += 1.0
        
        elif role == "seer" and action["type"] == "seer_divination":
            target = action["target"]
            if target in self.game_state["alive_players"]:
                # 记录查验结果（预言家视角）
                target_role = self.roles[target]
                self.game_state["action_history"].append(f"预言家查验玩家{target}为{target_role}")
                reward += 0.5
        
        elif role == "witch" and action["type"] in ["witch_heal", "witch_poison"]:
            target = action["target"]
            potion_type = action["type"].split("_")[1]
            
            if self.game_state["witch_potion"][potion_type] and target in self.game_state["alive_players"]:
                if action["type"] == "witch_heal":
                    self.game_state["action_history"].append(f"女巫救了玩家{target}")
                    reward += 0.6
                else:
                    self.game_state["action_history"].append(f"女巫毒了玩家{target}")
                    self.game_state["alive_players"].remove(target)
                    reward += 0.8
                self.game_state["witch_potion"][potion_type] = False
        
        # 切换到下一个玩家或白天
        next_player = current_player + 1
        if next_player >= self.num_players:
            self.game_state["phase"] = "day"
            self.game_state["current_player"] = 0
        else:
            self.game_state["current_player"] = next_player
        
        return self.get_state(), reward, False
    
    def _handle_day_action(self, action):
        """处理白天行动（讨论）"""
        current_player = self.game_state["current_player"]
        reward = 0.05  # 发言奖励
        
        if action["type"] == "discussion":
            # 添加讨论内容
            content = action.get("content", "无发言")  # 兼容空发言
            self.game_state["discussion_history"].append({
                "player": current_player,
                "content": content
            })
            # 额外奖励：发言长度合理的话加分
            if len(content) > 10:
                reward += 0.05
        
        # 切换到下一个玩家或投票阶段
        alive_players = self.game_state["alive_players"]

        # 修复：检查当前玩家是否在存活列表中
        if current_player not in alive_players:

    # 如果当前玩家已死亡，选择存活玩家列表中的第一个玩家
           if alive_players:               
               self.game_state["current_player"] = alive_players[0]
           return self.get_state(), reward, False

        next_player_idx = alive_players.index(current_player) + 1

        if next_player_idx >= len(alive_players):
            self.game_state["phase"] = "vote"
            self.game_state["current_player"] = alive_players[0] if alive_players else 0
        else:
            self.game_state["current_player"] = alive_players[next_player_idx]
            next_player_idx = alive_players.index(current_player) + 1
        
        if next_player_idx >= len(alive_players):
            self.game_state["phase"] = "vote"
            self.game_state["current_player"] = alive_players[0]
        else:
            self.game_state["current_player"] = alive_players[next_player_idx]
        
        return self.get_state(), reward, False
    
    def _handle_vote_action(self, action):
        """处理投票行动"""
        current_player = self.game_state["current_player"]
        reward = 0.1  # 投票奖励
        
        if action["type"] == "vote" and action["target"] in self.game_state["alive_players"]:
            target = action["target"]
            # 记录投票
            self.game_state["voting_history"].append(f"{current_player}->{target}")
            
            # 检查投票是否正确（投票狼人/好人，根据自身角色奖励）
            self_role = self.roles[current_player]
            target_role = self.roles[target]
            
            # 好人投狼人 / 狼人投好人 都加分
            if (self_role != "wolf" and target_role == "wolf") or (self_role == "wolf" and target_role != "wolf"):
                reward += 0.5
            # 投错扣分（避免乱投票）
            elif (self_role != "wolf" and target_role != "wolf") or (self_role == "wolf" and target_role == "wolf"):
                reward -= 0.2
        
        # 切换到下一个玩家或夜晚
        alive_players = self.game_state["alive_players"]
        if current_player not in alive_players:
             
    # 如果当前玩家已死亡，选择存活玩家列表中的第一个玩家
            if alive_players:
                self.game_state["current_player"] = alive_players[0]
            return self.get_state(), reward, False
        next_player_idx = alive_players.index(current_player) + 1
        
        if next_player_idx >= len(alive_players):
            # 计算投票结果
            self._calculate_vote_result()
            self.game_state["phase"] = "night"
            self.game_state["current_player"] = alive_players[0] if alive_players else 0
            self.game_state["round"] += 1
        else:
            self.game_state["current_player"] = alive_players[next_player_idx]
        
        return self.get_state(), reward, False
    
    def _calculate_vote_result(self):
        """计算投票结果"""
        votes = {}
        alive_players = self.game_state["alive_players"]
        # 只统计本轮存活玩家的投票
        recent_votes = self.game_state["voting_history"][-len(alive_players):]
        
        for vote in recent_votes:
            try:
                voter, target = vote.split("->")
                target = int(target)
                if target in alive_players:
                    votes[target] = votes.get(target, 0) + 1
            except:
                continue  # 兼容格式错误的投票记录
        
        if votes:
            # 找出得票最多的玩家
            max_votes = max(votes.values())
            eliminated_candidates = [t for t, v in votes.items() if v == max_votes]
            
            # 平局则随机淘汰一个（狼人杀规则）
            if len(eliminated_candidates) > 1:
                eliminated = random.choice(eliminated_candidates)
            else:
                eliminated = eliminated_candidates[0]
            
            if eliminated in alive_players:
                self.game_state["alive_players"].remove(eliminated)
                self.game_state["action_history"].append(f"投票淘汰玩家{eliminated}")
                
                # 猎人被投死触发开枪（狼人杀规则补充）
                if self.roles[eliminated] == "hunter":
                    hunter_target = random.choice([p for p in alive_players if p != eliminated])
                    self.game_state["alive_players"].remove(hunter_target)
                    self.game_state["action_history"].append(f"猎人带走玩家{hunter_target}")
    
    def _check_game_over(self):
        """检查游戏是否结束"""
        # 统计存活的狼人和好人数量
        alive_players = self.game_state["alive_players"]
        if not alive_players:
            self.game_state["game_over"] = True
            self.game_state["winner"] = "wolf"  # 无存活玩家默认狼人赢
            return
        
        alive_wolves = sum(1 for i in alive_players if self.roles[i] == "wolf")
        alive_good = len(alive_players) - alive_wolves
        
        # 游戏结束条件
        if alive_wolves == 0:
            self.game_state["game_over"] = True
            self.game_state["winner"] = "villager"  # 好人阵营赢
        elif alive_wolves >= alive_good:
            self.game_state["game_over"] = True
            self.game_state["winner"] = "wolf"  # 狼人阵营赢