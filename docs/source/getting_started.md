# Getting Started

This guide will walk you through installing TorchForge, understanding its dependencies, verifying your setup, and running your first training job.

## System Requirements

Before installing TorchForge, ensure your system meets the following requirements.

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **Operating System** | Linux (Fedora/Ubuntu/Debian) | MacOS and Windows not currently supported |
| **Python** | 3.10 or higher | Python 3.11 recommended |
| **GPU** | NVIDIA with CUDA support | AMD GPUs not currently supported |
| **Minimum GPUs** | 2+ for SFT, 3+ for GRPO | More GPUs enable larger models |
| **CUDA** | 12.8 | Required for GPU training |
| **RAM** | 32GB+ recommended | Depends on model size |
| **Disk Space** | 50GB+ free | For models, datasets, and checkpoints |
| **PyTorch** | Nightly build | Latest distributed features (DTensor, FSDP) |
| **Monarch** | Pre-packaged wheel | Distributed orchestration and actor system |
| **vLLM** | v0.10.0+ | Fast inference with PagedAttention |
| **TorchTitan** | Latest | Production training infrastructure |


## Prerequisites

- **Conda or Miniconda**: For environment management
  - Download from [conda.io](https://docs.conda.io/en/latest/miniconda.html)

- **GitHub CLI (gh)**: Required for downloading pre-packaged dependencies
  - Install instructions: [github.com/cli/cli#installation](https://github.com/cli/cli#installation)
  - After installing, authenticate with: `gh auth login`
  - You can use either HTTPS or SSH as the authentication protocol

- **Git**: For cloning the repository
  - Usually pre-installed on Linux systems
  - Verify with: `git --version`


**Installation note:** The installation script provides pre-built wheels with PyTorch nightly already included.

## Installation

TorchForge uses pre-packaged wheels for all dependencies, making installation faster and more reliable.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/meta-pytorch/forge.git
   cd forge
   ```

2. **Create Conda Environment**

   ```bash
   conda create -n forge python=3.10
   conda activate forge
   ```

3. **Run Installation Script**

   ```bash
   ./scripts/install.sh
   ```

   The installation script will:
   - Install system dependencies using DNF (or your package manager)
   - Download pre-built wheels for PyTorch nightly, Monarch, vLLM, and TorchTitan
   - Install TorchForge and all Python dependencies
   - Configure the environment for GPU training

   ```{tip}
   **Using sudo instead of conda**: If you prefer installing system packages directly rather than through conda, use:
   `./scripts/install.sh --use-sudo`
   ```

   ```{warning}
   When adding packages to `pyproject.toml`, use `uv sync --inexact` to avoid removing Monarch and vLLM.
   ```

## Verifying Your Setup

After installation, verify that all components are working correctly:

1. **Check GPU Availability**

   ```bash
   python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
   ```

   Expected output: `GPUs available: 2` (or more)

2. **Check CUDA Version**

   ```bash
   python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
   ```

   Expected output: `CUDA version: 12.8`
3. **Check All Dependencies**

   ```bash
   # Check core components
   python -c "import torch, forge, monarch, vllm; print('All imports successful')"

   # Check specific versions
   python -c "
   import torch
   import forge
   import vllm

   print(f'PyTorch: {torch.__version__}')
   print(f'TorchForge: {forge.__version__}')
   print(f'vLLM: {vllm.__version__}')
   print(f'CUDA: {torch.version.cuda}')
   print(f'GPUs: {torch.cuda.device_count()}')
   "
   ```

4. **Verify Monarch**

   ```bash
   python -c "
   from monarch.actor import Actor, this_host

   # Test basic Monarch functionality
   procs = this_host().spawn_procs({'gpus': 1})
   print('Monarch: Process spawning works')
   "
   ```

## Quick Start Examples

Now that TorchForge is installed, let's run some training examples.

Here's what training looks like with TorchForge:

```bash
# Install dependencies
conda create -n forge python=3.10
conda activate forge
git clone https://github.com/meta-pytorch/forge
cd forge
./scripts/install.sh

# Download a model
hf download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir /tmp/Meta-Llama-3.1-8B-Instruct --exclude "original/consolidated.00.pth"

# Run SFT training (requires 2+ GPUs)
uv run forge run --nproc_per_node 2 \
  apps/sft/main.py --config apps/sft/llama3_8b.yaml

# Run GRPO training (requires 3+ GPUs)
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

### Example 1: Supervised Fine-Tuning (SFT)

Fine-tune Llama 3 8B on your data. **Requires: 2+ GPUs**

1. **Download the Model**

   ```bash
   uv run forge download meta-llama/Meta-Llama-3.1-8B-Instruct \
     --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
     --ignore-patterns "original/consolidated.00.pth"
   ```

   ```{note}
   Model downloads require Hugging Face authentication. Run `huggingface-cli login` first if you haven't already.
   ```

2. **Run Training**

   ```bash
    python -m apps.sft.main --config apps/sft/llama3_8b.yaml
   ```

   **What's Happening:**
   - `--nproc_per_node 2`: Use 2 GPUs for training
   - `apps/sft/main.py`: SFT training script
   - `--config apps/sft/llama3_8b.yaml`: Configuration file with hyperparameters
   - **TorchTitan** handles model sharding across the 2 GPUs
   - **Monarch** coordinates the distributed training


   ```

### Example 2: GRPO Training

Train a model using reinforcement learning with GRPO. **Requires: 3+ GPUs**

```bash
python -m apps.grpo.main --config apps/grpo/qwen3_1_7b.yaml
```

**What's Happening:**
- GPU 0: Trainer model (being trained, powered by TorchTitan)
- GPU 1: Reference model (frozen baseline, powered by TorchTitan)
- GPU 2: Policy model (scoring outputs, powered by vLLM)
- **Monarch** orchestrates all three components
- **TorchStore** handles weight synchronization from training to inference

## Understanding Configuration Files

TorchForge uses YAML configuration files to manage training parameters. Let's examine a typical config:

```yaml
# Example: apps/sft/llama3_8b.yaml
model:
  name: meta-llama/Meta-Llama-3.1-8B-Instruct
  path: /tmp/Meta-Llama-3.1-8B-Instruct

training:
  batch_size: 4
  learning_rate: 1e-5
  num_epochs: 10
  gradient_accumulation_steps: 4

distributed:
  strategy: fsdp  # Managed by TorchTitan
  precision: bf16

checkpointing:
  save_interval: 1000
  output_dir: /tmp/checkpoints
```

**Key Sections:**
- **model**: Model path and settings
- **training**: Hyperparameters like batch size and learning rate
- **distributed**: Multi-GPU strategy (FSDP, tensor parallel, etc.) handled by TorchTitan
- **checkpointing**: Where and when to save model checkpoints

## Next Steps

Now that you have TorchForge installed and verified:

1. **Learn the Concepts**: Read {doc}`concepts` to understand TorchForge's architecture, including Monarch, Services, and TorchStore
2. **Explore Examples**: Check the `apps/` directory for more training examples
4. **Read Tutorials**: Follow {doc}`tutorials` for step-by-step guides
5. **API Documentation**: Explore {doc}`api` for detailed API reference

## Getting Help

If you encounter issues:

1. **Search Issues**: Look through [GitHub Issues](https://github.com/meta-pytorch/forge/issues)
2. **File a Bug Report**: Create a new issue with:
   - Your system configuration (output of diagnostic command below)
   - Full error message
   - Steps to reproduce
   - Expected vs actual behavior

**Diagnostic command:**
```bash
python -c "
import torch
import forge

try:
    import monarch
    monarch_status = 'OK'
except Exception as e:
    monarch_status = str(e)

try:
    import vllm
    vllm_version = vllm.__version__
except Exception as e:
    vllm_version = str(e)

print(f'PyTorch: {torch.__version__}')
print(f'TorchForge: {forge.__version__}')
print(f'Monarch: {monarch_status}')
print(f'vLLM: {vllm_version}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
"
```

Include this output in your bug reports!

## Additional Resources

- **Contributing Guide**: [CONTRIBUTING.md](https://github.com/meta-pytorch/forge/blob/main/CONTRIBUTING.md)
- **Code of Conduct**: [CODE_OF_CONDUCT.md](https://github.com/meta-pytorch/forge/blob/main/CODE_OF_CONDUCT.md)
- **Monarch Documentation**: [meta-pytorch.org/monarch](https://meta-pytorch.org/monarch)
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai)
- **TorchTitan**: [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan)
