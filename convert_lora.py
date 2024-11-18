from coreml_llama import Transformer, Llama_CoreML, ModelArgs
import torch
import argparse
import json
import gc
import sys

import coremltools as ct
import numpy as np

def load(args, path) -> Transformer:
    transformer = Transformer(args)

    model_pth = torch.load(path+"/consolidated.00.pth", map_location="cpu", weights_only=True)
    transformer.load_state_dict(model_pth, strict=False)
    transformer.eval()
    return transformer

def convert_model_to_coreml(model_dir, adapter_path, output_dir):
    # Conversion logic would go here
    print(f"Converting model from {model_dir} to Core ML format and saving to {output_dir}")
    # Core ML conversion code goes here


    with open(f"{model_dir}/params.json", "r") as st_json:
        params = json.load(st_json)
    args = ModelArgs(**params)
    transformer = load(args= args, path = model_dir)

    del args
    gc.collect()

    coreML_transformer = Llama_CoreML(transformer= transformer)
    coreML_transformer.transformer.load_state_dict(transformer.state_dict())

    del transformer
    gc.collect()

    input_ids: torch.Tensor = torch.zeros((1, 5), dtype=torch.int32)
    causal_mask: torch.Tensor = torch.zeros((1, 1, 5, 5), dtype=torch.float32)

    traced_transformer = torch.jit.trace(coreML_transformer.eval(),  [input_ids, causal_mask])

    caches_shape = coreML_transformer.transformer.caches_shape
    
    del coreML_transformer
    gc.collect()

    mlmodel = convert(traced_transformer, caches_shape)
    
    del traced_transformer
    gc.collect()

    mlmodel.save(output_dir)

def convert(traced, caches_shape):
    query_length = ct.RangeDim(lower_bound=1, upper_bound=2000, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=2000, default=1)
    inputs = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="input_ids"),
        ct.TensorType(shape=(1, 1, query_length, end_step_dim), dtype=np.int32, name="causal_mask"),
    ]

    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=caches_shape, 
                                    dtype=np.float16, 
                                    ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=caches_shape, 
                                    dtype=np.float16, 
                                    ),
            name="valueCache",
        ),
    ]

    outputs = [ct.TensorType(dtype=np.float16, name="logits")]


    mlmodel_fp16 = ct.convert(
        traced,
        inputs=inputs,
        states=states,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS18,
        # skip_model_load=True
    )

    return mlmodel_fp16

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Convert Llama3 model to Core ML format")
    
    # Adding arguments for model directory and output directory
    parser.add_argument('--model_dir', type=str, required=True, 
                        help="Path to the directory containing the Llama3 model weights.")
    parser.add_argument('--adapter_path', type=str, required=True, 
                        help="Path to the directory containing the Llama3 model weights.")
    parser.add_argument('--output_dir', type=str, default="llama3.mlpackage", 
                        help="Path to the directory where the converted Core ML model will be saved.")
    
    # Parsing arguments
    args = parser.parse_args()
    if not args.output_dir.endswith(".mlpackage"):
        print("Warning: It is recommended to use '.mlpackage' extension for Core ML output directories.")
        sys.exit(1)  # Exit the script with a non-zero exit code

    
    # Placeholder for conversion function
    convert_model_to_coreml(args.model_dir, args.adapter_path, args.output_dir)
    

if __name__ == "__main__":
    main()