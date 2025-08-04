#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to check whether a loaded model has tied lm_head and embedding weights.

This script checks if the input embedding layer and output language model head
share the same weight parameters, which is a common optimization in language models.
"""

import argparse
import sys

from transformers import AutoModel


def check_tied_weights(model):
    """
    Check if the model has tied weights between embedding and lm_head layers.

    Args:
        model: The loaded model to check

    Returns:
        tuple: (is_tied, embedding_param, lm_head_param, details)
    """
    embedding_param = None
    lm_head_param = None
    embedding_name = None
    lm_head_name = None

    # Common names for embedding layers
    embedding_patterns = [
        "embeddings.word_embeddings.weight",
        "embed_tokens.weight",
        "embedding.weight",
        "wte.weight",
        "transformer.wte.weight",
        "model.embed_tokens.weight",
    ]

    # Common names for lm_head layers
    lm_head_patterns = [
        "lm_head.weight",
        "embed_out.weight",
        "output.weight",
        "head.weight",
        "classifier.weight",
        "score.weight",
    ]  # Print all parameter names to help debug
    all_params = list(model.named_parameters())

    # Find embedding parameter - look for exact matches first, then partial matches
    for name, param in all_params:
        if any(pattern == name for pattern in embedding_patterns):
            embedding_param = param
            embedding_name = name
            break

    if embedding_param is None:
        # Try partial matches
        for name, param in all_params:
            embed_patterns = ["embed", "wte"]
            if any(
                pattern in name for pattern in embed_patterns if "embedding" in name.lower() or "embed" in name.lower()
            ):
                embedding_param = param
                embedding_name = name
                break

    # Find lm_head parameter - look for exact matches first, then partial matches
    for name, param in all_params:
        if any(pattern == name for pattern in lm_head_patterns):
            lm_head_param = param
            lm_head_name = name
            break

    if lm_head_param is None:
        # Try partial matches
        for name, param in all_params:
            head_patterns = ["lm_head", "head", "output", "classifier"]
            name_lower = name.lower()
            if any(
                pattern in name
                for pattern in head_patterns
                if "head" in name_lower or "output" in name_lower or "classifier" in name_lower
            ):
                lm_head_param = param
                lm_head_name = name
                break

    if embedding_param is None:
        param_names = [name for name, _ in all_params]
        return False, None, None, (f"Could not find embedding layer. Available parameters: {param_names[:10]}...")

    if lm_head_param is None:
        param_names = [name for name, _ in all_params]
        return (
            False,
            embedding_param,
            None,
            (f"Could not find lm_head layer. Available parameters: ...{param_names[-10:]}"),
        )

    # Check if they are the same parameter object
    is_tied = embedding_param is lm_head_param

    # Additional check: same data pointer
    if not is_tied and hasattr(embedding_param, "data") and hasattr(lm_head_param, "data"):
        is_tied = embedding_param.data.data_ptr() == lm_head_param.data.data_ptr()

    same_data_ptr = (
        embedding_param.data.data_ptr() == lm_head_param.data.data_ptr()
        if hasattr(embedding_param, "data") and hasattr(lm_head_param, "data")
        else False
    )

    details = {
        "embedding_name": embedding_name,
        "lm_head_name": lm_head_name,
        "embedding_shape": tuple(embedding_param.shape),
        "lm_head_shape": tuple(lm_head_param.shape),
        "same_object": embedding_param is lm_head_param,
        "same_data_ptr": same_data_ptr,
    }

    return is_tied, embedding_param, lm_head_param, details


def main():
    parser = argparse.ArgumentParser(description="Check if model has tied weights between embedding and lm_head")
    parser.add_argument("model_id", help="Model ID or path to check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    try:
        print(f"Loading model: {args.model_id}")

        # Load model - try different model classes to get the full model including LM head
        model = None
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
            print("Loaded with AutoModelForCausalLM")
        except Exception:
            try:
                from transformers import AutoModelForSeq2SeqLM

                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, trust_remote_code=True)
                print("Loaded with AutoModelForSeq2SeqLM")
            except Exception:
                try:
                    from transformers import AutoModelForMaskedLM

                    model = AutoModelForMaskedLM.from_pretrained(args.model_id, trust_remote_code=True)
                    print("Loaded with AutoModelForMaskedLM")
                except Exception:
                    # Fallback to AutoModel
                    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
                    print("Loaded with AutoModel (base model only, may not have LM head)")

        # Check tied weights
        is_tied, embedding_param, lm_head_param, details = check_tied_weights(model)

        # Check model config for tie_word_embeddings
        config_tied = getattr(model.config, "tie_word_embeddings", None)

        print(f"\nModel: {args.model_id}")
        print(f"Has tied weights: {is_tied}")
        if config_tied is not None:
            print(f"Config tie_word_embeddings: {config_tied}")
            if is_tied != config_tied:
                print("⚠️  Warning: Actual tied weights status differs from config!")

        if args.verbose and isinstance(details, dict):
            print("\nDetails:")
            print(f"  Embedding layer: {details['embedding_name']}")
            print(f"  LM head layer: {details['lm_head_name']}")
            print(f"  Embedding shape: {details['embedding_shape']}")
            print(f"  LM head shape: {details['lm_head_shape']}")
            print(f"  Same object: {details['same_object']}")
            print(f"  Same data pointer: {details['same_data_ptr']}")
        elif isinstance(details, str):
            print(f"Note: {details}")

        # Additional checks
        if is_tied and args.verbose:
            print("\n✓ Weights are tied - this model uses tied embeddings")
        elif not is_tied and embedding_param is not None and lm_head_param is not None:
            print("\n✗ Weights are NOT tied - embedding and lm_head use separate parameters")

    except Exception as e:
        print(f"Error loading model {args.model_id}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
