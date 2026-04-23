# CSC520 Project

This project simulates disaster-relief resource allocation across hubs and distress zones using a CSP + A* planning agent, with greedy and random baselines for comparison. It includes a matplotlib-based visualization so you can watch each strategy run on the same scenario.

Install dependencies:
`pip install -r requirements.txt`

You can optionally adjust scenario size in `scenario.py` (`NUM_HUBS`, `NUM_ZONES`, `NUM_TRUCKS`), then run:

Main demo visualization (CSP + A*):
`python main.py visualize`

Baseline visualizations:
- Greedy: `python main.py visualize greedy`
- Random: `python main.py visualize random`
