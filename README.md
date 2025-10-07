# 🛰️ Reinforcement Learning for Lunar Lander

## 📘 Project Overview
This project explores different reinforcement learning approaches for the **LunarLander-v3** environment in **Gymnasium (Box2D)**.  
We implement and compare a **hand-crafted heuristic baseline**, **Deep Q-Network (DQN)**, and **Proximal Policy Optimization (PPO)**.  
The goal is to evaluate **learning efficiency, stability, and robustness** under various conditions such as wind and turbulence.


## 🧩 Environment
- **State Space:** 8 continuous variables  
  *(position, velocity, angle, angular velocity, leg contact indicators)*  
- **Action Space:**  
  - Discrete(4): do nothing, fire left engine, fire main engine, fire right engine  
  - Continuous: throttle control for main and side engines  
- **Reward:** +100 for successful landing, −100 for crash, plus stepwise shaping rewards.


## ⚙️ Methods
| Method | Description |
|---------|--------------|
| **Heuristic Baseline** | Hand-designed controller based on position and velocity feedback. |
| **DQN** | Value-based Deep Reinforcement Learning for discrete actions. |
| **PPO** | Policy-gradient algorithm for stable learning in continuous control. |


## 📊 Evaluation Metrics
We evaluate the following metrics for each agent:
- **Average Reward (100 episodes)**
- **Landing Success Rate**
- **Crash Rate**
- **Energy Consumption Proxy** (engine usage penalty)
- **Landing Smoothness** (final position, velocity, and angle stability)


## 👥 Team Members
- Long Haitao (SJSU ID: 017413565)
- Jian Fu (SJSU ID: 018324436)
- Nikil Thalapaneni (SJSU ID: 019157047)


## 🧠 Advisor & Contact
- **Instructor:** Prof. Bertin Cordova Diba  
- **Emails:** bertin.cordovadiba@sjsu.edu  

- **Teaching Assistant:** Shanmukha Manoj Kakani
- **Emails:** shanmukhamanoj.kakani@sjsu.edu

