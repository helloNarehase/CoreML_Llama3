# Llama3 to Core ML Conversion Project

> This project aims to convert Meta’s Llama3 series models into Core ML’s stateful format for efficient execution on iOS or Mac-OS devices.


## Requirements
- Llama3 Series Weights: This project requires pretrained weights from Meta’s Llama3 series. The weights can be downloaded from the official Meta website.

## Key Features
- Llama3 Model Conversion to Core ML: Converts the model to a Core ML stateful structure.
- Model Architecture Modification: The architecture has been adjusted to ensure compatibility with Core ML.


## Usage

1. 	Download the Llama3 weights from Meta’s official website.
2.	Run the conversion script to generate a Core ML model.


## Installation & Execution

``` bash
# Install required libraries
pip install -r requirements.txt

# Run the conversion script
python convert_to_coreml.py --model_dir <path_to_downloaded_weights> --output_dir <path_to_save_model.mlpackage>

# exaple : python convert.py --model_dir ./Llama3.2-1B-Instruct --output_dir ./out.mlpackage 
```

## coreML - Quantization with INT4

``` bash
# Run the Quantization script
python coreml_quant.py --input_dir <path_to_input_model.mlpackage> --output_dir <path_to_save_quantized_model>

# exaple : python ./coreml_quant.py --input_dir ./out.mlpackage --output_dir ./quant_models
```


## This project is designed to work with the following models:
- Llama3.1 7B
- Llama3.2 1B
- Llama3.2 3B