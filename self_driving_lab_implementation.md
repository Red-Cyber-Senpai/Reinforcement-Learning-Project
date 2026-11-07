import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. Environment Definition
# ==========================================
class ChemicalReactorEnv:
    def __init__(self, budget=25):
        self.budget = budget
        self.reset()
        
    def reset(self):
        self.t = 0
        self.T = 25.0 # Starting ambient temperature
        self.C = 0.5  # Starting dilute concentration
        self.best_yield = 0.0
        return self._get_state()
        
    def _get_state(self):
        # State: [Normalized T, Normalized C, Remaining Budget Fraction]
        return np.array([self.T / 150.0, self.C / 2.0, (self.budget - self.t) / self.budget], dtype=np.float32)
        
    def step(self, action):
        # Action is [delta_T, delta_C] bounded between -1 and 1
        delta_T, delta_C = action
        
        # Physical clamp on operations
        self.T = np.clip(self.T + delta_T * 20.0, 20.0, 150.0)
        self.C = np.clip(self.C + delta_C * 0.5, 0.1, 2.0)
        
        # Simulated yield function (peaks around T=85, C=1.2)
        yield_val = np.exp(-((self.T - 85)**2)/1000 - ((self.C - 1.2)**2)/0.5)
        yield_val += np.random.normal(0, 0.02) # Gaussian sensor noise
        yield_val = np.clip(yield_val, 0, 1)
        
        # Composite Reward
        reward = yield_val
        if yield_val > self.best_yield:
            reward += (yield_val - self.best_yield) * 2.0 # Discovery bonus
            self.best_yield = yield_val
            
        reward -= self.C * 0.1 # Reagent cost penalty
        
        self.t += 1
        done = self.t >= self.budget
        
        return self._get_state(), reward, done

# ==========================================
# 2. Policy Network
# ==========================================
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(64, 2)
        # Learnable log standard deviation for exploration
        self.log_std = nn.Parameter(torch.zeros(2))
        
    def forward(self, x):
        features = self.fc(x)
        # Bounding the mean action between -1 and 1
        mean = torch.tanh(self.mean_head(features))
        std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=1.0))
        return mean, std

# ==========================================
# 3. Training Loop (REINFORCE)
# ==========================================
def train_agent():
    # Setup Device (MPS for Apple Silicon if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    env = ChemicalReactorEnv()
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    gamma = 0.99
    
    num_episodes = 500
    yield_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            mean, std = policy(state_tensor)
            
            # Sample continuous action
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Step environment
            action_np = action.cpu().squeeze().detach().numpy()
            next_state, reward, done = env.step(action_np)
            
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
                
        # Calculate standardized returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        
        yield_history.append(env.best_yield)
        
        # Logging
        if (episode + 1) % 50 == 0:
            rolling_avg = np.mean(yield_history[-50:])
            print(f"Episode {episode+1:3d}/500 | Episodic Best Yield: {env.best_yield:.4f} | "
                  f"50-Ep Avg Best: {rolling_avg:.4f} | Final T: {env.T:.1f}°C, C: {env.C:.2f}M")
            
if __name__ == "__main__":
    train_agent()