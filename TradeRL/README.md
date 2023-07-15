

---

# Reinforcement Learning Stock Trading(TradeRL)

This repository contains an implementation of a stock trading algorithm using reinforcement learning. The algorithm is trained using the Stable Baselines library and utilizes custom indicators and features to make trading decisions.


## Introduction

The goal of this project is to develop an automated trading system that leverages reinforcement learning techniques to make informed decisions on buying and selling stocks. The algorithm is trained on historical stock price data and learns to maximize profits by taking actions based on observed market conditions.

## Getting Started

### Prerequisites

To run the code in this repository, you need the following dependencies:

- Python 3.x
- gym
- gym_anytrading
- stable_baselines3
- finta
- quantstats
- pandas
- matplotlib

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your AlphaVantage API key to the script:

   ```python
   key = "your_api_key"
   ```

4. Run the code:

   ```bash
   python main.py
   ```

## Usage

The main script, `main.py`, demonstrates the entire process of training the reinforcement learning agent and evaluating its performance on a validation dataset. The script performs the following steps:

1. Fetches historical stock price data using the AlphaVantage API.
2. Preprocesses the data, including adding custom technical indicators.
3. Trains an A2C (Advantage Actor-Critic) agent using Stable Baselines.
4. Evaluates the trained agent on a validation dataset.
5. Plots the trading results and net worth over time.

Feel free to modify the script and experiment with different parameters, models, or data to further improve the trading performance.

## Results

After running the code, you will see the trading results plotted, including the net worth over time and other performance metrics. The key metric to note is the Compound Annual Growth Rate (CAGR), which measures the average annual return rate of the investment.

## Contributing

Contributions to this repository are welcome! If you have any ideas, suggestions, or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

You can customize and expand upon this template based on your specific implementation details and project goals.