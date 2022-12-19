import argparse
import logging
import os
import re
import string
from collections import Counter
from glob import glob
from pathlib import Path

import pdftotext

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level=10,
)


def parse_opts():
    parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--pdf", type=str, default=None, help="")

    opts = parser.parse_args()
    return opts


my_english_stop_words = [
    "a",
    "about",
    "above",
    "actually",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "also",
    "although",
    "always",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "became",
    "because",
    "become",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "either",
    "else",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "hence",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "let's",
    "may",
    "maybe",
    "me",
    "might",
    "mine",
    "more",
    "most",
    "must",
    "mustn't",
    "my",
    "myself",
    "neither",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "oh",
    "ok",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "whenever",
    "where",
    "where's",
    "whereas",
    "wherever",
    "whether",
    "which",
    "while",
    "who",
    "who's",
    "whoever",
    "whom",
    "whose",
    "why",
    "why's",
    "will",
    "with",
    "within",
    "without",
    "won't",
    "would",
    "wouldn't",
    "yes",
    "yet",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
]


def filter(bag):
    for word in my_english_stop_words:
        if word in bag:
            del bag[word]

    # filter numbers
    return bag


def should_keep(w):
    # filtering mechanism
    if not re.search("[a-zA-Z]", w):
        return False
    return True


# https://github.com/lhotse-speech/lhotse/blob/172e9f7a09e70cee4ee977fa9d288f96616cf7ec/lhotse/recipes/spgispeech.py
def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text


def process_word(w):
    # rules to process string w
    if w.endswith("TM"):
        w = w[:-2]
    if w.endswith("™"):
        w = w[:-1]
    w = w.lower()
    w = re.sub(
        "^([^a-zA-Z0-9]*)(.*?)([^a-zA-Z0-9]*)$", r"\2", w
    )  # remove heading and trailing symbols
    w = re.sub("’", "", w)
    w = normalize(w)
    return w.strip()


def get_bag_of_words_(lines):
    bag = Counter()
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        # words = line.split()
        words = re.split(r"\s|/", line)
        for w in words:
            if should_keep(w):
                w = process_word(w)
                if len(w) > 0:
                    bag[w] += 1
    return bag


def get_bag_of_words_context(filename):
    bag = Counter()

    with open(filename, "rb") as f:
        pdf = pdftotext.PDF(f)

    pages = []
    for page in pdf:
        page = page.lower()
        if "forward looking" in page or "forward-looking" in page:
            # logging.info("skip 1 page..")
            continue
        pages.append(page)

    pdf_content = "\n\n".join(pages)
    bag_ = get_bag_of_words_(pdf_content.split("\n"))
    bag_ = filter(bag_)
    bag += bag_

    return bag


def main(opts):

    context_bag = get_bag_of_words_context(opts.pdf)
    # logging.info(f"len(context_bag)={len(context_bag)}")
    # logging.info(f"context_bag={context_bag}")
    for w, c in context_bag.items():
        print(w)


if __name__ == "__main__":
    opts = parse_opts()

    main(opts)
