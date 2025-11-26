# BATAK-game-playtesters-analysis

This repository contains a full analysis of playtester behaviour for the Batak card game, including raw data, processed outputs, a player dataset (.json), and the model code used to evaluate player performance and decision patterns.

The goal of this project is to understand how players make decisions in Batak, identify skill differences, and produce a data-driven evaluation of gameplay patterns.

Project Description

This project focuses on analysing gameplay data collected from Batak playtesters.
The dataset includes:

Player identifiers

Cards played

Trick results

Bids and bid accuracy

Round outcomes

Strategy selections

Optional metadata (experience level, playstyle tags, etc.)

Using this data, the project applies modelling techniques to evaluate:

Player performance

Strategy consistency

Optimal vs. suboptimal decisions

Predictive modelling for game outcomes

Patterns in card selection and bidding behaviour

🧠 Analysis Overview

The analysis explores:

✔ 1. Player Behaviour

How each player bids

Aggressive vs conservative strategies

Consistency in trick-taking performance

✔ 2. Performance Metrics

Win rate

Bid accuracy

Trick success ratio

Round-to-round stability

✔ 3. Model Insights

The provided model code evaluates:

Player decision-making scores

Predictions of trick outcomes

Strategy clustering (e.g., risk-taker, balanced, defensive)

Game flow simulations using extracted player profiles

✔ 4. Visual Output

In the /output/ folder you may find:

Accuracy charts

Player comparison graphs

Histograms of bid distribution

Heatmaps of card-play patterns

Summary tables
