# Evaluating vision–language models in multi-turn pragmatic interpretation

This repository contains code and materials for the manuscript **"Which one is banana man? Evaluating vision–language models in multi-turn pragmatic interpretation"**. 

## Repository structure
```
├── context_prep
│   ├── full_feedback
│   ├── human_yoked
│   └── practice
├── src
├── tests
├── scripts
├── data
│   └── images
└── analysis_scripts
```

1. `context_prep` contains the script needed to generate trial structures to be used for model evaluation
2. `src` contains scripts for model evaluation, with tests in `tests` and shell scripts for launching in `scripts`
3. `data` stores the grid image, the span annotations, and logprobs; logprobs should be downloaded from [OSF](https://osf.io/zk8gq/overview).
4. `analysis_scripts` contains R scripts for analysis, used in conjunction with the Quarto documents in the root folder.