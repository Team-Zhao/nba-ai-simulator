# NBA AI Simulator

An end-to-end NBA game prediction system built around dynamic player ratings, roster-based team features, and machine learning.

The project converts historical NBA player performance into six-dimensional player ratings, aggregates those ratings into team-level matchup features, and predicts both game outcomes and expected point differential.

## V1 MVP

V1 establishes the complete modeling pipeline:

```text
Historical Player Data
        ↓
Player Rating Engine
        ↓
Expected-Minutes Team Aggregation
        ↓
Game Matchup Features
        ↓
Machine Learning Models
        ↓
Win Probability + Predicted Margin
