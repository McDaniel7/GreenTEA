# GreenTEA

GreenTEA is an auto-prompting framework that leverages genetic algorithms and large language models (LLMs) to optimize prompts for knowledge-intensive tasks. It's designed to evolve prompts through iterations and gradually improve their effectiveness on specific tasks.

### Features

- Genetic algorithm-based prompt evolution
- Gradient-guided optimization
- LLMs for prompt generation and evaluation

## Table of Contents
1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [Usage](#usage)
4. [Output](#output)
5. [Components](#components)
6. [Contribution](#contribution)
7. [Disclaimer](#disclaimer)

## Installation

To install GreenTEA, clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Package Structure

```
GreenTEA/
├── config_generation/
│   ├── __init__.py
│   ├── config.yaml
│   └── generate_config.py
├── data/
│   ├── __init__.py
│   └── data_processor.py
├── models/
│   ├── __init__.py
│   ├── investigator.py
│   ├── prompt_generator.py
│   └── error_hunter.py
├── utils/
│   ├── __init__.py
│   ├── ga_utils.py
│   ├── llm_utils.py
│   ├── text_utils.py
│   ├── preprocessing.py
│   ├── text_cleaning.py
│   └── scoring.py
├── config.py
├── main.py
├── README.md
└── requirements.txt
```

## Usage

GreenTEA can be run in two steps:

1. Generate a configuration file:
   ```bash
   cd config_generation
   python generate_config.py config.yaml
   ```
   This will create a JSON configuration file based on the parameters specified in `config.yaml`.

2. Run the main GreenTEA algorithm:
   ```bash
   cd ..
   python main.py --config path/to/saved/config.json
   ```
   This will execute the GreenTEA algorithm using the generated configuration.

#### Example configuration YAML:

GreenTEA uses a JSON configuration file. Here's an example of the configuration structure:

```json
{
    "task": "GSM8K",
    "dataset": "main",
    "dev_file": "path/to/dev/file",
    "test_file": "path/to/test/file",
    "model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
    "pop_size": 5,
    "budget": 20,
    "gradient_guided": true,
    "output_path": "path/to/output",
    "sbert_model_path": "path/to/sbert/model",
    "prompt_template_path": "path/to/prompt/template",
    "crossover_prompt_path": "path/to/crossover/prompt",
    "error_hunter_prompt_path": "path/to/error/hunter/prompt"
}
```

For a full list of configuration options, please refer to the `Config` class in `config.py`.

#### Example GreenTEA usage in ```main.py```:

```python
from config import Config
from data.data_processor import DataPreparer
from models.investigator import LLMInvestigator
from models.prompt_generator import LLMGuidedCrossoverChildGenerator
from models.error_hunter import LLMErrorHunter
from utils.scoring import AccEvaluator
from utils.llm_utils import LLMPromptRephraser
from utils.ga_utils import Evolutor
from utils.text_utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description='GreenTEA prompt optimization')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Load configuration
config = Config.from_json(args.config)

# Initialize components
data_preparer = DataPreparer(config.answer_is_label)
evaluator = AccEvaluator()
investigator = LLMInvestigator(config.prompt_template_path, config.model_name)
prompt_generator = LLMGuidedCrossoverChildGenerator(config.crossover_prompt_path, config.model_name)
prompt_rephraser = LLMPromptRephraser(config.rephraser_template_path, config.model_name)
error_hunter = LLMErrorHunter(config.error_hunter_prompt_path, config.model_name)

# Initialize evolutor
evolutor = Evolutor(
    config=config,
    data_preparer=data_preparer,
    investigator=investigator,
    evaluator=evaluator,
    prompt_generator=prompt_generator,
    prompt_paraphraser=prompt_rephraser,
    error_hunter=error_hunter
)

# Run evolution
evolutor.evolute(manual_prompt_file_path=config.manual_prompt_file_path,
                 llm_prompt_file_path=config.llm_prompt_file_path)
```

## Output

GreenTEA algorithm produces several outputs:

1. **Evolved Prompts**: The best-performing prompts after the evolution process.
2. **Performance Metrics**: Scores for each generation and the final best-performing prompt.
3. **Log Files**: Detailed logs of the evolution process, including intermediate results.
4. **Configuration File**: A JSON file containing all parameters used for the run.

Example output structure:
```
output/
├── config_timestamp.json
├── evol.log
├── step1_pop_experiment_name.txt
├── step2_pop_experiment_name.txt
...
├── stepN_pop_experiment_name.txt
└── stepN_pop_test_experiment_name.txt
```

## Components

- **Evolutor**: The main class that orchestrates the genetic algorithm process.
- **DataPreparer**: Prepares the data for evaluation.
- **AccEvaluator**: Evaluates the model performance.
- **LLMInvestigator**: Uses LLMs to investigate and generate responses for given prompts.
- **LLMGuidedCrossoverChildGenerator**: Generates new prompts by combining and mutating existing ones.
- **LLMErrorHunter**: Analyzes errors in the generated outputs to guide optimization.
- **LLMPromptRephraser**: Rephrases prompts to introduce variety in the population.

## Contribution

Contributions to GreenTEA are welcome!

## Disclaimer

This project is for research purposes only. Ensure you comply with the terms of service of any third-party APIs or services used.
