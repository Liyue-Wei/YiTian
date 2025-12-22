import copy

class LandmarkSmoother:
    def __init__(self, alpha=0.5):
        """
        初始化平滑器
        :param alpha: 平滑係數 (0.0 ~ 1.0)。
                      值越大，越接近當前檢測值 (反應快，但抖動大)；
                      值越小，越依賴上一幀的值 (越平滑，但會有延遲)。
                      0.5 是一個平衡點。
        """
        self.alpha = alpha
        self.prev_landmarks = None

    def process(self, current_landmarks):
        """
        對關鍵點進行平滑處理
        :param current_landmarks: 當前幀檢測到的關鍵點列表 (MediaPipe multi_hand_landmarks 對象)
        :return: 平滑後的關鍵點列表
        """
        # 1. 如果當前沒有檢測到手，重置歷史記錄
        if not current_landmarks:
            self.prev_landmarks = None
            return None

        # 2. 如果是第一幀，或者手部數量發生變化（例如從單手變雙手），
        #    無法與上一幀對應，直接使用當前值並初始化歷史記錄。
        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current_landmarks):
            self.prev_landmarks = copy.deepcopy(current_landmarks)
            return current_landmarks

        # 3. 執行平滑算法
        try:
            # 遍歷每一隻手
            for i in range(len(current_landmarks)):
                hand = current_landmarks[i]
                prev_hand = self.prev_landmarks[i]
                
                # 遍歷該手的所有關鍵點 (MediaPipe 手部模型有 21 個點)
                for j in range(len(hand.landmark)):
                    lm = hand.landmark[j]          # 當前幀的點
                    prev_lm = prev_hand.landmark[j] # 上一幀平滑後的點
                    
                    # === 核心算法公式 ===
                    # NewValue = alpha * Current + (1 - alpha) * Previous
                    lm.x = self.alpha * lm.x + (1 - self.alpha) * prev_lm.x
                    lm.y = self.alpha * lm.y + (1 - self.alpha) * prev_lm.y
                    lm.z = self.alpha * lm.z + (1 - self.alpha) * prev_lm.z
            
            # 4. 更新歷史記錄
            # 必須使用 deepcopy，否則 prev_landmarks 會變成指向 current_landmarks 的引用，
            # 導致下一幀計算時 Previous 和 Current 永遠相同，失去平滑效果。
            self.prev_landmarks = copy.deepcopy(current_landmarks)
            
        except (IndexError, AttributeError) as e:
            print(f"平滑處理錯誤: {e}")
            self.prev_landmarks = None # 出錯時重置
            
        return current_landmarks