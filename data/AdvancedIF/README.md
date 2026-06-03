---
license: cc-by-nc-4.0
language:
- en
tags:
- instruction following
- multi-turn
- LLM
- rubric-based
---

### Dataset Summary

We introduce AdvancedIF, a new benchmark featuring over 1,600 prompts and expert-curated rubric designed to assess LLMs' proficiency in 
* Complex instruction following: each prompt has 6+ instructions with combination
of one, format, style, structure, length, negative constraints, spelling, and inter-conditional instructions;
* Multi-turn instruction following: the ability to follow instruction carried from previous;
* System prompt steerability: The ability to follow instructions in the system prompt.


See paper for full details: 

[AdvancedIF: Rubric-Based Benchmarking and Reinforcement Learning for Advancing LLM Instruction Following](https://arxiv.org/abs/2511.10507)


### Evaluation Script
https://github.com/facebookresearch/AdvancedIF


### Data Splits

* test: 1,645 examples