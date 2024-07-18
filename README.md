# Deep Q-Learning Player for the Microservice Dungeon Game

This repository implements a Deep Q-Learning algorithm for the [Microservice Dungeon](https://www.archi-lab.io/compounds/dungeon_main.html) game, which was a main part of my bachelor thesis.

## Overview

The Microservice Dungeon is a complex game environment designed to challenge AI agents. This project aims to create an intelligent agent capable of playing the game effectively using Deep Q-Learning, a popular reinforcement learning technique.

## Key Features

- Implementation of a Deep Q-Learning algorithm
- Custom environment integration with the Microservice Dungeon game
- Accelerated training using a [custom Game Server written in Rust](https://github.com/Amueller36/MSD-Game-Service)
- Detailed experimentation and performance analysis

## Quickstart

1. Start the Custom Game Server
2. Install the requirements.txt for your local Python Environment.
3. Start the Training via python main_dqn.py

## Experiments and Results

I conducted several experiments to evaluate the performance of our Deep Q-Learning agent. Below are video demonstrations of key experiments:

### Experiment 1: Trained Agent vs. Itself
[![Experiment 1](http://img.youtube.com/vi/7O5fXNGslSk/0.jpg)](http://www.youtube.com/watch?v=7O5fXNGslSk "Trained Agent playing vs itself")

### Experiment 2: Trained Agent (Exp. 2) vs. Trained Agent (Exp. 1)
[![Experiment 2](http://img.youtube.com/vi/1KngxE6auoo/0.jpg)](http://www.youtube.com/watch?v=1KngxE6auoo "Trained Agent of Experiment 2 playing vs Trained Agent of Experiment 1")

### Experiment 3: Trained Agent (Exp. 1) vs. Random Acting Agent
[![Experiment 3](http://img.youtube.com/vi/n17dhXWLjfg/0.jpg)](http://www.youtube.com/watch?v=n17dhXWLjfg "Trained Agent of Experiment 1 playing vs Random Acting Agent")
