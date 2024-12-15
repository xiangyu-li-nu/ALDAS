# Adaptive Longitudinal Driving Assistance System with Reinforcement Learning

This repository contains the codebase for the paper **"An Adaptive Longitudinal Driving Assistance System with Reinforcement Learning"**, authored by Xiangyu Li, Pengyuan Liu, Hani S. Mahmassani, and Ying Chen. The work introduces a novel Adaptive Longitudinal Driving Assistance System (ALDAS) that leverages reinforcement learning (specifically, the Deep Deterministic Policy Gradient (DDPG) algorithm) to optimize car-following behavior in autonomous vehicles. It integrates human driving preferences to improve driver acceptance of automated systems.

## Abstract

The proposed system employs a cyber-physical approach by integrating CARLA and SUMO simulators, alongside a driving simulator, to better align automated driving systems with human driving behavior. The framework optimizes car-following performance by balancing safety, efficiency, comfort, and fuel consumption. Experimental results demonstrate the effectiveness of ALDAS, which outperforms traditional models such as LSTM, RNN, and CACC in simulations involving diverse traffic conditions and autonomous vehicle penetration rates.

## File Descriptions

### **Project Directory Structure**

- **`CARLA1_G29_NEW.py`**
  - This script integrates CARLA with a Logitech G29 steering wheel for simulation purposes. It sets up a driving environment in CARLA, including vehicle spawn points, sensor configurations, and control mechanisms. Users can toggle between manual and autonomous driving modes during the simulation.

- **`DDPG_CF.py`**
  - The primary script for implementing the DDPG-based car-following model. It integrates SUMO and CARLA to simulate mixed-traffic environments and provides functionalities for synchronization between the two simulators. Includes training configurations for the DDPG model and methods for co-simulation.

- **`wheel_config.ini`**
  - Configuration file for mapping the Logitech G29 steering wheel inputs to the CARLA simulator.

- **`/DDPG_CF` Directory**
  - Contains modules and dependencies for DDPG training and SUMO-CARLA integration:
    - `agent/`: Implements reinforcement learning agents for training.
    - `data/`: Stores datasets generated during simulations.
    - `examples/`: Example configurations and workflows for running simulations.
    - `results/`: Output files and logs from simulations.
    - `src/`: Source code for training, evaluating, and reporting results.
    - `sumo_integration/`: Code for connecting SUMO and CARLA simulators.

- **`/carla/` Directory**
  - CARLA-specific helper functions and examples for simulation tasks.

- **`examples/` Directory**
  - Demonstrates various simulation setups and use cases for testing the car-following model.

- **`util/` Directory**
  - Utility functions for data processing, configuration management, and debugging.

## Contact

**Author**: Xiangyu Li  
**Email**: [xiangyuli2027@u.northwestern.edu](mailto:xiangyuli2027@u.northwestern.edu)  
**Affiliation**: Second-year Ph.D. student, Transportation Engineering, Northwestern University.

For further questions or discussions, feel free to reach out!
