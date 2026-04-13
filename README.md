  <img width="300" height="300" alt="Programmauswahl für Transformer" src="https://github.com/user-attachments/assets/623cba78-c703-4dce-9136-6446813d1f45" />

# Inference-Time Program Selection for Transformer Models

This repository explores **adaptive execution of transformer layers at inference time**.

Instead of running a transformer with a fixed layer order, we treat the sequence of layers as a **program**. A model can execute different layer sequences depending on the input, potentially improving performance without retraining the base network.

The repository provides tools to:

* **Search for effective layer programs** using Monte Carlo Tree Search (MCTS)
* **Learn routers** that predict which program to execute for a given input
* **Represent routing decisions as modular deviations** from a default transformer
* **Evaluate routing policies** across benchmarks and tasks

Conceptually, the system consists of three components:

1. **Flexible Transformer Execution**
   Transformer layers can be skipped, repeated, or reordered. This turns the fixed architecture into a space of executable programs.

2. **Program Search**
   MCTS is used to discover high-performing layer sequences for tasks or benchmarks.

3. **Routing Networks**
   Lightweight neural routers predict which program (or program deviations) to execute for a given input.

The goal is to study whether **inference-time architectural adaptation** can improve reasoning, generalization, and efficiency in large language models.
