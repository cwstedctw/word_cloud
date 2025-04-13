# Chinese Word Cloud Generator with CKIP Tagger

This project creates Chinese word clouds from text data using CKIP Tagger for text analysis. CKIP Tagger is a powerful tool developed by Academia Sinica for Chinese text processing, including word segmentation, part-of-speech tagging, and named entity recognition.

## System Requirements

- **Operating System**: Windows
- **Python Version**: 3.7.6

## Quick Setup Guide

### Step 1: Download & Install

1. Clone this repository:
   ```bash
   git clone https://github.com/cwstedctw/word_cloud.git
   cd ckiptagger
   ```

2. Download TensorFlow (Required):
   - **Important**: The TensorFlow wheel file is not included in the repository
   - Download TensorFlow 1.13.1 (AVX2 version) from [this link](https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.13.1/py37/CPU/avx2)
   - Save the downloaded `.whl` file (tensorflow-1.13.1-cp37-cp37m-win_amd64.whl) in your project directory

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Prepare Your Data

1. Make sure the `model/` directory contains all required model files
   - This directory should already include the word segmentation, part-of-speech tagging, and named entity recognition models

2. Place your input data in the `information/` directory
   - Required files:
     - Excel files with your text data (e.g., `110-01test.xlsx`)
     - `remove.csv`: Words to exclude from analysis
     - `synonym.csv`: Synonyms to combine during analysis
   
   Note: The `110_01/` subdirectory included in the repository is just a demo for a specific semester. You should place your own data files directly in the `information/` directory.

### Step 3: Run the Program

```bash
python main.py
```

Generated word clouds will be saved in the `word_cloud/` directory.

## Directory Structure

- `model/`: Contains pre-trained CKIP models
- `information/`: Place your input files here (the included `110_01/` is just a demo)
- `word_cloud/`: Output directory for generated word clouds
- `data.zip`: Additional resources (extract if needed)
- `help_function.py`: Utility functions
- `config.py`: Configuration settings

## Advanced Usage

To customize the word cloud generation:
1. Modify `synonym.csv` to group related terms together
2. Update `remove.csv` to exclude specific words from the analysis
3. Adjust settings in `config.py` to change processing parameters

## Additional Resources

- [CKIP Tagger Documentation](https://github.com/ckiplab/ckiptagger) - Learn more about the underlying text analysis tools
- If you encounter issues with TensorFlow installation, ensure you're using the specific version required (1.13.1 AVX2)
