# 🧠 CogniPulse  
### *Driving Decision Dynamics under DBS Modulation*

---

## 📌 About This Project  

**CogniPulse** is a computational framework for studying how **neural dynamics influence decision-making** in healthy and Parkinsonian brain states.

It integrates:

- 🔹 A **biophysical spiking STN–GPe network**, and  
- 🔹 A **Basal Ganglia decision-making model** (Iowa Gambling Task)
---

## 🔍 Scientific Motivation  

Basal Ganglia circuits play a central role in:

- action selection  
- reinforcement learning  
- exploration–exploitation balance  

In Parkinson’s disease, abnormal **synchrony and oscillatory activity**—especially in the **STN–GPe loop**—lead to:

- impaired decision-making  
- reduced cognitive flexibility  

**CogniPulse** enables systematic investigation of:

- how pathological neural dynamics alter behavior  
- how DBS modulates neural states  
- how noisy DBS stimulation influence decision outcomes  

---

## 📚 Conceptual Basis  

This framework is inspired by:

> Mandali et al. (2015)  
> *A spiking Basal Ganglia model of synchrony, exploration and decision making*  
> Frontiers in Neuroscience  

---

## 🧩 Repository Structure  

The project is modular and designed for clarity and extensibility:

### 📁 `basal_ganglia/`  
Implements reinforcement learning mechanisms and decision policies for the Iowa Gambling Task.

---

### 📁 `envs/`  
Contains the **Iowa Gambling Task (IGT)** environment used for behavioral simulations.

---

### 📁 `stn_gpe/`  
Core spiking neural model of the **STN–GPe circuit**, including:

- network simulation  
- DBS signal generation  
- neural metrics:
  - synchrony  
  - entropy  
  - firing rate  
  - spectral analysis  

---

### 📁 `params/`  
YAML-based configuration system for:

- neural regimes (Normal / PD / DBS)  
- stimulation parameters  
- decision model settings  

---

### 📁 `simulations/`  
Jupyter notebooks demonstrating:

- STN–GPe neural dynamics  
- decision-making performance (IGT)  
Interactive interfaces are included for running simulations without modifying code.

---

## ⚙️ Installation  

Clone the repository:

```bash
git clone https://github.com/CharithaPalika/CogniPulse.git
cd CogniPulse