"""Data preparation script for the real book continual learning benchmark.

Downloads public domain books, trains a subword BPE tokenizer, splits the book text
into sequential chunks, and generates keyword-matched evaluation prompts and fact probes.
"""

from __future__ import annotations
import argparse
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Standard public-domain mirror URLs (cache.epub URLs are highly reliable)
OZ_URL = "https://www.gutenberg.org/cache/epub/55/pg55.txt"
ALICE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
TIME_MACHINE_URL = "https://www.gutenberg.org/cache/epub/35/pg35.txt"

# Curated QA Prompt Pool for Wizard of Oz with keyword mapping for chunk assignment
OZ_PROMPTS_POOL = [
    {
        "keywords": ["Kansas", "Dorothy", "Aunt Em", "Uncle Henry", "Toto"],
        "question": "Question: Where does Dorothy live? Answer:",
        "answer": "Kansas",
        "group": "early_book_prompts"
    },
    {
        "keywords": ["Toto", "dog", "black"],
        "question": "Question: What is the name of Dorothy's dog? Answer:",
        "answer": "Toto",
        "group": "early_book_prompts"
    },
    {
        "keywords": ["Munchkins", "Witch of the East", "shoes"],
        "question": "Question: Who are the small people Dorothy meets first in Oz? Answer:",
        "answer": "Munchkins",
        "group": "early_book_prompts"
    },
    {
        "keywords": ["Scarecrow", "cornfield", "brains"],
        "question": "Question: Who does Dorothy rescue from a pole in the cornfield? Answer:",
        "answer": "Scarecrow",
        "group": "early_book_prompts"
    },
    {
        "keywords": ["Tin Woodman", "oil", "heart"],
        "question": "Question: What material is the rusted woodman made of? Answer:",
        "answer": "tin",
        "group": "middle_book_prompts"
    },
    {
        "keywords": ["Lion", "Cowardly Lion", "courage"],
        "question": "Question: What does the Cowardly Lion want from the Wizard? Answer:",
        "answer": "courage",
        "group": "middle_book_prompts"
    },
    {
        "keywords": ["Emerald City", "Green", "Wizard"],
        "question": "Question: Where does the Wizard of Oz live? Answer:",
        "answer": "Emerald City",
        "group": "middle_book_prompts"
    },
    {
        "keywords": ["Wicked Witch", "melted", "water"],
        "question": "Question: What destroys the Wicked Witch of the West? Answer:",
        "answer": "water",
        "group": "late_book_prompts"
    },
    {
        "keywords": ["Glinda", "Witch of the South"],
        "question": "Question: Who is the Good Witch of the South? Answer:",
        "answer": "Glinda",
        "group": "late_book_prompts"
    },
    {
        "keywords": ["Kansas", "shoes", "home"],
        "question": "Question: What color shoes does Dorothy wear to go home? Answer:",
        "answer": "silver",
        "group": "late_book_prompts"
    },
    # Cross-chunk composition prompts
    {
        "keywords": ["Scarecrow", "Tin Woodman"],
        "question": "Question: Who did Dorothy meet first, the Scarecrow or the Tin Woodman? Answer:",
        "answer": "Scarecrow",
        "group": "cross_chunk_prompts"
    },
    {
        "keywords": ["brains", "heart", "courage"],
        "question": "Question: What did the Scarecrow want, and what did the Tin Woodman want? Answer:",
        "answer": "brains and heart",
        "group": "cross_chunk_prompts"
    }
]

# Curated Fact Probes for Wizard of Oz
OZ_FACTS_POOL = [
    ("Dorothy lives in Kansas with Aunt Em and Uncle Henry.", ["Kansas", "Dorothy"]),
    ("Toto is Dorothy's small black dog.", ["Toto", "dog"]),
    ("The house landed on the Wicked Witch of the East.", ["house", "East"]),
    ("The Munchkins are small people who live in the East.", ["Munchkins"]),
    ("The Witch of the North gave Dorothy the silver shoes.", ["shoes", "North"]),
    ("The Scarecrow is made of straw and wants brains.", ["Scarecrow", "brains"]),
    ("The Tin Woodman is made of tin and wants a heart.", ["Tin Woodman", "heart"]),
    ("The Cowardly Lion wants courage and joins the group.", ["Lion", "courage"]),
    ("The Emerald City is ruled by the Wizard of Oz.", ["Emerald", "Wizard"]),
    ("The Wicked Witch of the West is melted by water.", ["Wicked Witch", "water"]),
    ("Glinda is the Good Witch of the South.", ["Glinda", "South"]),
    ("Dorothy claps the silver shoes together to go home.", ["shoes", "home"])
]

def _validate_gutenberg_download(text: str, desc: str) -> None:
    if "END OF THE PROJECT GUTENBERG EBOOK" not in text.upper():
        raise RuntimeError(f"Downloaded text for {desc} is missing the Project Gutenberg end marker.")


def download_book(url: str, desc: str, cache_path: Path, retries: int, timeout_seconds: float) -> str:
    if cache_path.exists():
        cached = cache_path.read_text(encoding="utf-8")
        _validate_gutenberg_download(cached, desc)
        print(f"Loaded cached {desc} from {cache_path}")
        return cached

    print(f"Downloading {desc} from {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    )
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                text = response.read().decode("utf-8")
            _validate_gutenberg_download(text, desc)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
            return text
        except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, RuntimeError) as error:
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if attempt < retries:
                time.sleep(1.5 * attempt)
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to download a complete copy of {desc} after {retries} attempts.\n{joined}")


def _looks_like_toc_entry(line: str) -> bool:
    value = line.strip()
    if not value:
        return False
    if value.lower().startswith(("chapter ", "introduction", "epilogue")):
        return True
    return re.match(r"^(?:[IVXLCDM]+|\d+)[\.\s]+[A-Z]", value) is not None


def strip_table_of_contents(text: str) -> str:
    lines = text.splitlines()
    contents_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "contents":
            contents_idx = idx
            break
    if contents_idx is None:
        return text

    saw_toc_entry = False
    blank_run = 0
    end_idx = None
    for idx in range(contents_idx + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped:
            blank_run = 0
            if _looks_like_toc_entry(stripped):
                saw_toc_entry = True
            continue
        blank_run += 1
        if saw_toc_entry and blank_run >= 2:
            end_idx = idx + 1
            while end_idx < len(lines) and not lines[end_idx].strip():
                end_idx += 1
            break

    if end_idx is None:
        raise RuntimeError("Found a Contents section but could not identify its end.")
    return "\n".join(lines[:contents_idx] + [""] + lines[end_idx:])

def clean_gutenberg_text(text: str) -> str:
    start_markers = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"END OF THE PROJECT GUTENBERG EBOOK",
    ]
    
    start_idx = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            start_idx = match.end()
            break
            
    end_idx = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_idx = match.start()
            break
            
    cleaned = strip_table_of_contents(text[start_idx:end_idx].strip())
    return cleaned

def train_tokenizer(texts: list[str], vocab_size: int, output_path: Path) -> None:
    print(f"Training BPE Tokenizer with vocab_size={vocab_size}...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        vocab_size=vocab_size
    )
    # Temporary file to train on
    tmp_path = output_path.parent / "tmp_train.txt"
    tmp_path.write_text("\n\n".join(texts), encoding="utf-8")
    
    tokenizer.train([str(tmp_path)], trainer)
    tokenizer.save(str(output_path))
    
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Tokenizer saved to {output_path}")

def chunk_text(text: str, num_chunks: int) -> list[str]:
    words = text.split()
    chunk_size = len(words) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(words)
        chunks.append(" ".join(words[start:end]))
    return chunks

def assign_prompts_and_facts(chunks: list[str]) -> tuple[list[dict], dict[str, list[str]], dict[str, list[dict]]]:
    # Group prompts for evaluation
    eval_groups = {
        "early_book_prompts": [],
        "middle_book_prompts": [],
        "late_book_prompts": [],
        "cross_chunk_prompts": []
    }
    
    chunk_data = []
    fact_probes = {}
    
    # Track assigned indices to avoid duplicates
    assigned_prompts = set()
    assigned_facts = set()
    
    for idx, chunk_text_content in enumerate(chunks):
        chunk_id = f"chunk_{idx+1:02d}"
        local_prompts = []
        
        # 1. Match non-composition prompts to chunks.
        for p_idx, prompt in enumerate(OZ_PROMPTS_POOL):
            if p_idx in assigned_prompts:
                continue
            if prompt["group"] == "cross_chunk_prompts":
                continue
            if all(kw.lower() in chunk_text_content.lower() for kw in prompt["keywords"]):
                p_item = {"question": prompt["question"], "answer": prompt["answer"]}
                local_prompts.append(p_item)
                eval_groups[prompt["group"]].append(p_item)
                assigned_prompts.add(p_idx)
                
        # 2. Match Facts to Chunks
        local_facts = []
        for f_idx, (fact, keywords) in enumerate(OZ_FACTS_POOL):
            if f_idx in assigned_facts:
                continue
            if all(kw.lower() in chunk_text_content.lower() for kw in keywords):
                local_facts.append(fact)
                assigned_facts.add(f_idx)
                
        fact_probes[chunk_id] = local_facts
        chunk_data.append({
            "chunk_id": chunk_id,
            "text": chunk_text_content,
            "local_prompts": local_prompts,
            "retention_prompts": [],
            "composition_prompts": []
        })
        
    eval_groups["cross_chunk_prompts"] = [
        {"question": prompt["question"], "answer": prompt["answer"]}
        for prompt in OZ_PROMPTS_POOL
        if prompt["group"] == "cross_chunk_prompts"
    ]

    # Missing assignments mean the benchmark is malformed. Do not silently place
    # prompts into arbitrary chunks.
    missing_prompts = [
        prompt["question"]
        for p_idx, prompt in enumerate(OZ_PROMPTS_POOL)
        if prompt["group"] != "cross_chunk_prompts" and p_idx not in assigned_prompts
    ]
    missing_facts = [
        fact
        for f_idx, (fact, _) in enumerate(OZ_FACTS_POOL)
        if f_idx not in assigned_facts
    ]
    if missing_prompts or missing_facts:
        raise RuntimeError(
            "Could not assign all prompts/facts to book chunks.\n"
            f"missing_prompts={missing_prompts}\n"
            f"missing_facts={missing_facts}"
        )

    # Populate retention and cross-chunk prompts for each chunk
    all_local_prompts = []
    for idx, c_data in enumerate(chunk_data):
        c_data["retention_prompts"] = [p for p in all_local_prompts]
        all_local_prompts.extend(c_data["local_prompts"])
        
        # Cross-chunk composition prompts are evaluated at late stages (chunk >= N/2)
        if idx >= len(chunks) // 2:
            c_data["composition_prompts"] = [
                {"question": p["question"], "answer": p["answer"]}
                for p in OZ_PROMPTS_POOL
                if p["group"] == "cross_chunk_prompts"
            ]
            
    return chunk_data, fact_probes, eval_groups

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare data files for real book continual learning benchmark.")
    parser.add_argument("--num-chunks", type=int, default=5, help="Number of text chunks for Wizard of Oz.")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Vocab size for tokenizer training.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real_book"), help="Directory to save data.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/real_book"), help="Directory to save tokenizer.")
    parser.add_argument("--download-retries", type=int, default=4, help="Number of attempts for each Gutenberg download.")
    parser.add_argument("--download-timeout", type=float, default=30.0, help="Seconds before a Gutenberg request times out.")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download and clean books
    raw_cache_dir = args.output_dir / "raw_cache"
    oz_raw = download_book(
        OZ_URL,
        "The Wonderful Wizard of Oz",
        raw_cache_dir / "wizard_of_oz_raw.txt",
        retries=args.download_retries,
        timeout_seconds=args.download_timeout,
    )
    alice_raw = download_book(
        ALICE_URL,
        "Alice's Adventures in Wonderland",
        raw_cache_dir / "alice_raw.txt",
        retries=args.download_retries,
        timeout_seconds=args.download_timeout,
    )
    tm_raw = download_book(
        TIME_MACHINE_URL,
        "The Time Machine",
        raw_cache_dir / "time_machine_raw.txt",
        retries=args.download_retries,
        timeout_seconds=args.download_timeout,
    )
    
    oz_clean = clean_gutenberg_text(oz_raw)
    alice_clean = clean_gutenberg_text(alice_raw)
    tm_clean = clean_gutenberg_text(tm_raw)
    
    # Save raw clean texts
    (args.output_dir / "book.txt").write_text(oz_clean, encoding="utf-8")
    (args.output_dir / "alice.txt").write_text(alice_clean, encoding="utf-8")
    (args.output_dir / "time_machine.txt").write_text(tm_clean, encoding="utf-8")
    print("Clean book texts saved successfully.")
    
    # 2. Train subword BPE Tokenizer
    train_tokenizer(
        [oz_clean, alice_clean, tm_clean],
        args.vocab_size,
        args.checkpoint_dir / "tokenizer.json"
    )
    # Save a copy in data directory as well
    import shutil
    shutil.copy(args.checkpoint_dir / "tokenizer.json", args.output_dir / "tokenizer.json")
    
    # 3. Create sequential chunk splits, QA prompts, and fact probes
    chunks = chunk_text(oz_clean, args.num_chunks)
    chunk_data, fact_probes, eval_groups = assign_prompts_and_facts(chunks)
    
    # Save output data files
    with open(args.output_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)
        
    with open(args.output_dir / "eval_prompts.json", "w", encoding="utf-8") as f:
        json.dump(eval_groups, f, indent=2)
        
    with open(args.output_dir / "fact_probes.json", "w", encoding="utf-8") as f:
        json.dump(fact_probes, f, indent=2)
        
    print(f"Data preparation complete! Created {args.num_chunks} chunks.")
    print(f"Saved: chunks.json, eval_prompts.json, fact_probes.json in {args.output_dir}")

if __name__ == "__main__":
    main()
