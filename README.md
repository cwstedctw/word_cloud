# CKIP Tagger Project

This project uses the CKIP Tagger for Chinese text analysis. CKIP Tagger is a neural network-based Chinese word segmentation, part-of-speech tagging, and named entity recognition tool developed by Academia Sinica.

## Requirements

- Python 3.7.6
- Required packages listed in `requirements.txt`
- Specific TensorFlow wheel file (see Installation)

## Installation

1.  Clone this repository.
2.  **Download TensorFlow:** This project requires a specific version of TensorFlow (`1.13.1` for Python 3.7, CPU with AVX2 support). Download the wheel file from the following link and place it in the root directory of this project (where `requirements.txt` is located):
    [https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.13.1/py37/CPU/avx2/tensorflow-1.13.1-cp37-cp37m-win_amd64.whl](https://github.com/fo40225/tensorflow-windows-wheel/blob/master/1.13.1/py37/CPU/avx2/tensorflow-1.13.1-cp37-cp37m-win_amd64.whl)
    Make sure the downloaded file is named `tensorflow-1.13.1-cp37-cp37m-win_amd64.whl`.
3.  **Clone `word_cloud` Dependency:** Clone the necessary `word_cloud` project repository:
    ```bash
    git clone https://github.com/cwstedctw/word_cloud.git
    ```
4.  **Install required packages:** Once the TensorFlow wheel file is in place and the `word_cloud` repository is cloned, run the following command:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

[Add usage instructions here]

## Notes

- This project requires specific model files that need to be downloaded separately.
- See the [CKIP Tagger documentation](https://github.com/ckiplab/ckiptagger) for more details.

## License

[Add license information]
