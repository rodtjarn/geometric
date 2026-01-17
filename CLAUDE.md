# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains educational materials on the intersection of category theory, geometric deep learning, and causal machine learning. The content is designed for learning and research purposes.

### Content

- **research_guide.md**: Comprehensive implementation guide covering geometric deep learning, causal ML architectures, and category theory in ML. Includes Python code examples for PyTorch Geometric, e3nn, DoWhy, and EconML frameworks.

- **category_theory_journey.html**: Interactive web-based tutorial that builds from foundational mathematics (sets, functions, algebra) through group theory and category theory to causal reasoning in ML. Contains JavaScript-driven interactive exercises.

## Architecture Notes

The HTML file is a self-contained single-page application with:
- Embedded CSS styling (gradient backgrounds, responsive layout)
- Vanilla JavaScript for interactive components (matrix multiplication demos, function composition exercises)
- Tab-based navigation through 8 learning levels (Overview → Sets & Functions → Algebra → Groups → Geometry → Categories → Causal ML → State-of-Art)
- Progress tracking via visual progress bar

## Getting Started

### View the Interactive Tutorial

Open `category_theory_journey.html` in a browser:
```bash
xdg-open category_theory_journey.html  # Linux
open category_theory_journey.html      # macOS
```

### Python Environment Setup

The research guide contains runnable Python code examples. Install dependencies:

```bash
# Core ML framework
pip install torch

# Geometric deep learning
pip install torch-geometric
pip install e3nn
pip install escnn

# Causal inference
pip install dowhy
pip install econml
```

### Running Code Examples

Code snippets from `research_guide.md` can be copied and run directly. The complete molecule property prediction example (lines 588-729) demonstrates E(n)-equivariant graph neural networks using QM9 dataset.

## Key Frameworks Referenced

| Framework | Purpose |
|-----------|---------|
| PyTorch Geometric | Graph neural networks, equivariant networks |
| e3nn | E(3)-equivariant operations for molecules/physics |
| escnn | Steerable CNNs for continuous rotation equivariance |
| DoWhy | Causal inference with do-calculus |
| EconML | Heterogeneous treatment effect estimation |
