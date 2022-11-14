from __future__ import annotations

import argparse
import glob
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
import tokenize
import warnings
from dataclasses import dataclass
from token import COMMENT, INDENT, STRING
from typing import Optional

import fasttext
import tqdm
from torch.utils.data import Dataset
from transformers import pipeline

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class SimpleDataset(Dataset):
    data: list[str]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> str:
        return self.data[index]


def remove_comments_from_code(code: str) -> str:
    output, last_token, last_lineno, last_column = "", INDENT, -1, 0
    token_iterator = tokenize.generate_tokens(io.StringIO(code).readline)

    try:
        for token, text, (slineno, scolumn), (elineno, ecolumn), _ in token_iterator:
            if scolumn > last_column:
                indent = scolumn if slineno > last_lineno else scolumn - last_column
                output += " " * indent
            # Extract a token text which is not comment or docstring.
            if token != COMMENT and (token != STRING or last_token != INDENT):
                output += text
            last_token, last_lineno, last_column = token, elineno, ecolumn
    except Exception:
        pass
    return output


def preprocess_chunked_notebooks(
    filenames: list[str],
    notebook_dir: str,
    output_dir: str,
    fasttext_model: Optional[str],
    return_queue: mp.Queue,
):
    model = fasttext.load_model(fasttext_model) if fasttext_model else None
    for filename in filenames:
        with open(os.path.join(notebook_dir, filename)) as fp:
            notebook = json.load(fp)

        # Remove comments from python code in code cells. Note that other languages
        # (e.g. HTML) can be written with `%%` magic, so the code cells which start with
        # `%%` will not be processed.
        for name, code in notebook["source"].items():
            if notebook["cell_type"][name] == "code" and not code.startswith("%%"):
                notebook["source"][name] = remove_comments_from_code(code)

        # Predict the language of the notebook markdown contents.
        lang = "en"
        if model is not None:
            markdown_content = ""
            for name, cell in notebook["source"].items():
                if notebook["cell_type"][name] == "markdown":
                    markdown_content += " ".join(cell.split()) + " "
            lang = model.predict(markdown_content)[0][0].replace("__label__", "")

        # Write the preprocessed notebook to the json file on the output directory.
        if not os.path.exists(os.path.join(output_dir, lang)):
            os.makedirs(os.path.join(output_dir, lang), exist_ok=True)
        with open(os.path.join(output_dir, lang, filename), "w") as fp:
            json.dump(notebook, fp)

        return_queue.put(False)
    return_queue.put(True)


def preprocess_notebooks_with_multiprocesses(args: argparse.Namespace):
    # Create multi-processes to preprocess the notebook codes and save to the output
    # directory.
    processes, return_queue, filenames = [], mp.Queue(), os.listdir(args.notebook_dir)
    for i in range(args.num_cores):
        params = (
            filenames[i :: args.num_cores],
            args.notebook_dir,
            args.output_dir,
            None if args.no_translate else args.fasttext_model,
            return_queue,
        )
        process = mp.Process(
            target=preprocess_chunked_notebooks, args=params, daemon=True
        )
        process.start()
        processes.append(process)

    # Wait until the processes are exited with updating a progress bar.
    with tqdm.tqdm(filenames) as tbar:
        num_completed = 0
        while num_completed < args.num_cores:
            if return_queue.get():
                num_completed += 1
            else:
                tbar.update()


def translate_notebooks(args: argparse.Namespace):
    translator = pipeline("translation", model=args.translation_model, device=0)
    translator.tokenizer.model_max_length = args.max_length

    for lang in os.listdir(args.output_dir):
        if not os.path.isdir(os.path.join(args.output_dir, lang)):
            continue
        if lang not in translator.tokenizer.lang_code_to_id or lang == "en":
            continue

        notebooks = {}
        for filename in os.listdir(os.path.join(args.output_dir, lang)):
            with open(os.path.join(args.output_dir, lang, filename)) as fp:
                notebooks[filename] = json.load(fp)

        markdowns, cell_names = [], []
        for filename, notebook in notebooks.items():
            for cell_name, cell in notebook["source"].items():
                if notebook["cell_type"][cell_name] == "markdown":
                    markdowns.append(cell)
                    cell_names.append((filename, cell_name))

        # For efficient prediction, the markdown contets should be sorted to make the
        # texts in same batch have almost same lengths.
        indices = sorted(enumerate(markdowns), key=lambda x: len(x[1]))
        markdowns = [markdowns[i] for i, _ in indices]
        cell_names = [cell_names[i] for i, _ in indices]

        translated = translator(
            SimpleDataset(markdowns),
            src_lang=lang,
            tgt_lang="en",
            batch_size=args.batch_size,
            max_length=args.max_length,
            truncation=True,
            num_beams=args.num_beams,
            temperature=args.temperature,
            early_stopping=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        translated = [
            output[0]["translation_text"]
            for output in tqdm.tqdm(translated, desc=lang, total=len(markdowns))
        ]

        # Overwrite the markdown cells with their tranlsated texts.
        for content, (filename, cell_name) in zip(translated, cell_names):
            notebooks[filename]["source"][cell_name] = content
        for filename, notebook in notebooks.items():
            with open(os.path.join(args.output_dir, "en", filename), "w") as fp:
                json.dump(notebook, fp)
        shutil.rmtree(os.path.join(args.output_dir, lang))


def main(args: argparse.Namespace):
    preprocess_notebooks_with_multiprocesses(args)
    if not args.no_translate:
        translate_notebooks(args)

    for filename in glob.glob(os.path.join(args.output_dir, "*/*.json")):
        shutil.move(filename, os.path.join(args.output_dir, os.path.basename(filename)))
    for filename in os.listdir(args.output_dir):
        if os.path.isdir(os.path.join(args.output_dir, filename)):
            shutil.rmtree(os.path.join(args.output_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-dir", default="resources/ai4code/train")
    parser.add_argument("--output-dir", default="resources/ai4code/train_cleaned")
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    parser.add_argument("--fasttext-model", default="resources/lid.176.ftz")
    parser.add_argument("--translation-model", default="facebook/m2m100_418M")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--no-translate", default=False, action="store_true")
    main(parser.parse_args())
