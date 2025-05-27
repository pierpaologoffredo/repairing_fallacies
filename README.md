# Repairing Fallacious Argumentation in Political Debates

## Project Overview

This repository contains the code and data for the paper "Repairing Fallacious Argumentation in Political Debates" by Pierpaolo Goffredo, Deborah Dore, Serena Villata, and Elena Cabrio.

## Abstract

This project addresses the challenging task of repairing fallacious arguments in political debates. The main contributions are:

1. A novel dataset, **FallacyFix**, comprising 747 repaired examples across various fallacy categories.
2. Modular prompt techniques for generating non-fallacious arguments, both dependent and independent of the specific fallacy label.
3. A rigorous evaluation methodology to assess the accuracy of generated non-fallacious arguments.
4. A human evaluation of the generated non-fallacious arguments to assess their acceptability.

## Dataset: FallacyFix

- Source: Based on the ElecDeb60to20-fallacy dataset of U.S. presidential debates (1960-2020)
- Fallacy categories: Appeal to Fear, Appeal to Pity, Appeal to Popular Opinion, Flag Waving, and Loaded Language
- Two versions: 
  - Standard version: 747 repaired annotations
  - Compact version: 541 examples (paraphrasing or partial text modification only)

## Methodology

- Task: Repairing fallacious arguments in political debates
- Approach: Using Large Language Models (LLMs) to generate repaired arguments
- Settings: Zero-Shot, Few-Shot, and Fine-Tuning
- Models tested: BART, Google Gemma, Mistral 7B, Mixtral 8x7B, LLaMa 3, OpenAI GPT, Claude
- Configurations: With/without fallacy label and context

## Results

<!-- ### Fallacy Classification (Macro F1 Score)

| Model | Context Only (CO) | No Fallacy Label & Context (NO) |
|-------|-------------------|--------------------------------|
| BART (FT) | 34.92% | 42.20% |
| Claude 3 (ZS) | 37.94% | 18.55% |
| GPT-4 (FS) | 59.15% | 37.69% |
| Llama 3 8B (FT) | 80.46% | 88.64% |

### Argument Repair (Best BERTScore results)

| Setting | Context Only (CO) | Label & Context (LC) | No Context (NO) | Label Only (LO) |
|---------|-------------------|----------------------|-----------------|-----------------|
| Zero-Shot | 0.69 (GPT-4) | 0.71 (GPT-4) | 0.62 (GPT-3.5) | 0.60 (GPT-4) |
| Few-Shot | 0.71 (Claude 3) | 0.78 (Claude 3) | 0.69 (GPT-3.5) | 0.72 (Gemma 7B) |
| Fine-Tuning | 0.98 (BART) | 0.98 (BART) | 0.96 (Llama 3) | 0.97 (Llama 3) | -->

### Human Evaluation

- Relevance: 4.03 ± 0.68
- Suitability: 4.17 ± 0.68
- Cogency: 3.76 ± 0.69

(Scores on a 5-point Likert scale)

## Key Findings

1. LLMs show promise in repairing fallacious arguments, with fine-tuned models performing best.
2. Including fallacy labels and context generally improves performance.
3. Human evaluators found the repaired arguments relevant and suitable, but slightly less cogent.
4. There's still room for improvement in logical coherence and adherence to prompt instructions.


<!-- ## Citation

If you use this work, please cite:

```
@inproceedings{trovato2025repairing,
  title={Repairing Fallacious Argumentation in Political Debates},
  author={Trovato, Ben and Tobin, G.K.M. and Th{\o}rv{\"a}ld, Lars},
  booktitle={Proceedings of ACM SAC Conference (SAC'25)},
  year={2025},
  organization={ACM}
}
```

## License

(Add license information) -->

## Contact

For any questions or issues, please open an issue in this repository or contact the authors.
