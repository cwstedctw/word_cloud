# Chinese Word Cloud Generator with CKIP Tagger (Windows)

Welcome! This project helps you create cool word clouds from Chinese text using the powerful CKIP Tagger for analysis. CKIP Tagger, developed by Academia Sinica, is great for tasks like splitting sentences into words (segmentation), identifying parts of speech (POS tagging), and finding named entities (NER).

**Let's get you set up to generate some word clouds!**

## System Requirements

*   **Operating System**: **Windows Only**
*   **Python Version**: 3.7.6 (Make sure you have this version installed)
*   **Recommended Editor**: Visual Studio Code (VS Code) - We'll use it for setup!

## Quick Setup Guide

### Step 0: Set Up Your Coding Environment (VS Code + Virtual Environment)

Using a virtual environment is highly recommended to keep project dependencies tidy. We'll use VS Code's built-in terminal for this.

1.  **Install VS Code**: If you don't have it, download and install VS Code from [here](https://code.visualstudio.com/).
2.  **Install Python**: Make sure you have Python 3.7.6 installed for Windows. You can get it from the [official Python website](https://www.python.org/downloads/release/python-376/). Remember to check "Add Python 3.7 to PATH" during installation.
3.  **Open Project in VS Code**:
    *   Clone or download this project first (see Step 1 below if you haven't).
    *   Open VS Code.
    *   Go to `File` > `Open Folder...` and select the `ckiptagger` directory you just cloned/downloaded.
4.  **Create Virtual Environment**:
    *   Open the terminal in VS Code (`Terminal` > `New Terminal`).
    *   In the terminal, type the following command and press Enter:
        ```bash
        python -m venv .venv
        ```
        This creates a folder named `.venv` containing a private Python environment.
5.  **Activate Virtual Environment**:
    *   In the *same* VS Code terminal, run:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   You should see `(.venv)` appear at the beginning of your terminal prompt, indicating it's active.
    *   VS Code might also ask if you want to select this environment for the workspace - click "Yes".
6.  **Keep this terminal open** for the next steps!

### Step 1: Get the Project Code & TensorFlow

1.  **Clone the Repository** (if you haven't already): Open a *new* terminal or use Git Bash outside VS Code if you prefer:
    ```bash
    git clone https://github.com/cwstedctw/word_cloud.git
    ```
    *Then navigate into the folder:*
    ```bash
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

