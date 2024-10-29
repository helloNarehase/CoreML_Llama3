import argparse
import os
import coremltools as ct

BLUE = "\033[94m"
RESET = "\033[0m"

def load_model(model_path):
    # Load the model from the specified .mlpackage file
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"{model_path} does not exist.")
    print(f"{BLUE}Model loaded from: {model_path}{RESET}")
    
    return ct.models.MLModel(model_path)

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Quantize Core ML model")
    
    # Adding arguments for input and output directories
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Path to the .mlpackage model file.")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the directory where the quantized model will be saved.")
    
    # Parsing arguments
    args = parser.parse_args()
    
    # Load the model from the input .mlpackage file
    mlmodel_fp16 = load_model(args.input_dir)
    
    # Block-wise quantize model weights to int4
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=[1, 32],
    )
    
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the quantized model
    output_model_path = os.path.join(args.output_dir, "quantized_model.mlpackage")
    mlmodel_int4.save(output_model_path)
    print(f"Quantized model saved to {BLUE}{output_model_path}{RESET}")

if __name__ == "__main__":
    main()