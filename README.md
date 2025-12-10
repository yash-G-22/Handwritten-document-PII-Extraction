# Handwritten OCR with PII Detection

This project implements an end-to-end OCR (Optical Character Recognition) pipeline specifically designed for handwritten documents, with built-in PII (Personally Identifiable Information) detection and redaction capabilities.

## Features

- Image preprocessing for better OCR accuracy
- Automatic deskewing of tilted documents
- Text extraction using Tesseract OCR
- PII detection for common patterns (emails, phone numbers, SSNs, etc.)
- Text redaction to mask sensitive information
- Support for multiple input formats (JPEG, PNG)
- Command-line interface for batch processing

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR engine installed on your system
   - **Windows**: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. Place your input images in the `handwritten_test` directory
2. Run the pipeline:
   ```bash
   python ocr_pipeline.py
   ```
3. Check the `output` directory for results

### Output Files

For each processed image, the following files will be generated:
- `preprocessed_<original_filename>.jpg`: Preprocessed image
- `results_<original_filename>.json`: Detailed processing results
- `redacted_<original_filename>.txt`: Text with PII redacted

### Command Line Arguments

You can specify custom input and output directories:

```bash
python ocr_pipeline.py --input /path/to/input --output /path/to/output
```

## Customization

### Adding New PII Patterns

To add new PII detection patterns, modify the `PII_PATTERNS` dictionary in the `PIIDetector` class in `ocr_pipeline.py`.

### Tuning OCR Parameters

You can adjust Tesseract parameters in the `OCRProcessor.extract_text` method for better recognition results.

## Troubleshooting

- **Tesseract not found**: Ensure Tesseract is installed and in your system PATH
- **Poor OCR results**: Try adjusting the preprocessing steps in `ImagePreprocessor`
- **Memory issues**: Process large documents one at a time

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Tesseract OCR engine
- OpenCV for image processing
- Pytesseract for Python bindings
