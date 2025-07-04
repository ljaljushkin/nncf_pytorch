from pathlib import Path
from transformers import AutoModelForCausalLM
import argparse
import sys
import json
from pprint import pprint
import torch
from lm_eval import simple_evaluate
from lm_eval.models.optimum_lm import HFLM


def dump_results(results, save_dir, task_name):
    pprint(results["results"])
    results["config"]["model_dtype"] = str(results["config"]["model_dtype"])
    save_file = save_dir / f'results_{task_name}_777.json'
    print(f'save to {save_file}')
    with save_file.open('w') as f:
        json.dump(results, f, indent=4)



def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)

    # Model params
    parser.add_argument(
        "-f", "--folder",
        type=str,
    )
    parser.add_argument(
        "-m", "--model_id",
        type=str,
    )
    parser.add_argument(
        "-s", "--seqlen",
        type=int,
        default=4096
    )
    parser.add_argument(
        "-n", "--no-chat",
        action="store_true",
    )

    return parser

def main(argv) -> float:
    parser = get_argument_parser()
    args = parser.parse_args(argv)
    pprint(vars(args))

    name = args.folder
    for idx in [1]:
        ckpt_dir = Path(name) / 'last'
        print("#" * 50 + f" Evaluate Reference" + "#" * 50)
        model_id = args.model_id
        apply_chat_template=not args.no_chat
        assert (model_id == 'mistralai/Mistral-7B-v0.3') != apply_chat_template

        device = 'cuda'
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

        task = "gsm8k"
        lm_obj = HFLM(pretrained=model, batch_size=16)
        results = simple_evaluate(lm_obj, tasks=[task], log_samples=False, apply_chat_template=apply_chat_template)
        dump_results(results, ckpt_dir, task)
        del lm_obj

        lm_obj = HFLM(pretrained=model, max_length=args.seqlen)
        task = "wikitext"
        results = simple_evaluate(lm_obj, tasks=[task], log_samples=False)
        dump_results(results, ckpt_dir, task)
        pprint(results["results"])
        del lm_obj

        lm_obj = HFLM(pretrained=model, batch_size=16)
        task = "hellaswag"
        results = simple_evaluate(lm_obj, tasks=[task], log_samples=False)
        dump_results(results, ckpt_dir, task)
        del lm_obj

if __name__ == "__main__":
    main(sys.argv[1:])