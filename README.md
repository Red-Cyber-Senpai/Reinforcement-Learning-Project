# 🧪 Self-Driving Lab Simulator: Autonomous Synthesis Optimization

An autonomous chemical synthesis optimization agent built with Reinforcement Learning. This project implements the **REINFORCE (Monte Carlo Policy Gradient)** algorithm in PyTorch to maximize simulated reaction yield within a strict 25-step experimental budget. 

## 📋 Project Overview
Discovering optimal chemical reaction conditions traditionally relies on resource-intensive manual trial-and-error. This project frames reaction tuning as a **Continuous Markov Decision Process (MDP)**, allowing an RL agent to autonomously navigate a complex, non-linear optimization landscape with Gaussian sensor noise.

The agent successfully converges to the theoretical maximum yield (100.0%) by dynamically adjusting temperature and chemical concentration, while balancing a discovery bonus and reagent cost penalties.

## 🧠 Problem Formulation (MDP)
* **State Space ($S \in \mathbb{R}^3$)**: 
  * Normalized Temperature ($20^\circ\text{C}$ to $150^\circ\text{C}$)
  * Normalized Concentration ($0.1\text{M}$ to $2.0\text{M}$)
  * Remaining experimental budget fraction
* **Action Space ($A \in \mathbb{R}^2$)**: Continuous adjustments to $\Delta \text{Temperature}$ and $\Delta \text{Concentration}$, sampled from a Gaussian policy network.
* **Reward Function ($R$)**: Composite score rewarding immediate yield, a discovery bonus for breakthrough maximums, and a penalty for excessive reagent consumption.
* **Constraints**: Fixed budget of 25 steps per episode.

## 🚀 Key Features
* **PyTorch REINFORCE Implementation**: Custom Multi-Layer Perceptron (MLP) outputting mean ($\mu$) and log standard deviation ($\sigma$) for continuous action sampling.
* **Hardware Acceleration**: Automatic device placement supporting Apple Silicon Metal Performance Shaders (`mps`) and `cpu` fallback.
* **Custom Environment**: A lightweight, dependency-free chemistry lab simulator built in NumPy.

## 📊 Results & Convergence
The agent was trained for 500 episodes ($12,500$ total steps). 
* **Global Optimum Discovered**: $100.0\%$ Yield
* **Optimal Conditions Found**: $T = 83.72^\circ\text{C}, C = 1.21\text{M}$ (Theoretical ground truth: $85.0^\circ\text{C}, 1.20\text{M}$)
* **Stability**: The 50-episode moving average best yield steadily converged from $0.7595 \to 0.9477$.


<img width="3300" height="1800" alt="learning_curve_plot" src="https://github.com/user-attachments/assets/dcbab1a7-7e9d-4025-94a6-49b02135c9d9" />


## ⚙️ Installation & Usage

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/self-driving-lab-rl.git
cd self-driving-lab-rl
```

**2. Install dependencies**

```bash
pip install torch numpy matplotlib
```

**3. Run the training pipeline**

```bash
python train.py
```

Alternatively, you can run the provided Jupyter Notebook (`Project7_Self_Driving_Lab.ipynb`) for an interactive breakdown of the environment and training loop.

## 📁 Repository Structure

```text
├── env.py                          # Continuous MDP chemical reactor environment
├── agent.py                        # PyTorch REINFORCE policy network
├── train.py                         # Main training loop and Matplotlib plotting
├── Project7_Self_Driving_Lab.ipynb  # Interactive Jupyter Notebook version
├── screenshots/                     # Generated learning curves and terminal logs
└── README.md
```

## 👨‍💻 Team

| Name | Roll No. |
|---|---|
| Saketcharan Chauhan M | 100523729054 |
| Abubakr Siddiq Mohammed | 100523729063 |
