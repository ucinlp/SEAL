"""Process the Pile dataset into an FM-Index."""
import argparse
import glob
import io
import json
import logging
from urllib import parse
from typing import List

import boto3
import transformers
from tqdm import tqdm

from seal.index import FMIndex
from seal.vocab import Vocab, character_tokenizer


logger = logging.getLogger(__name__)

S3_PREFIX = 's3://'
TOKENIZER = transformers.AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')


def label_enumerator(prefix, iterable):
    for i, line in enumerate(iterable):
        yield f'{prefix}-{i}', line


def s3_iterator(input_files: str):
    """Iterates over an s3 bucket of text files."""
    logger.info('Using S3 iterator')
    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects_v2')
    parsed_uri = parse.urlparse(input_files)
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    logger.info('Bucket: %s', bucket)
    logger.info('Key: %s', key)
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=key)
    for page in page_iterator:
        for result in page['Contents']:
            obj = client.get_object(Bucket=bucket, Key=result['Key'])
            yield from label_enumerator(result['Key'],
                                        io.TextIOWrapper(obj['Body']))


def file_iterator(input_files: str):
    """Iterates over text files matching a pattern."""
    logger.info('Using file iterator.')
    for fname in glob.glob(input_files):
        with open(fname, 'r') as f:
            yield from label_enumerator(fname, f)


def input_iterator(input_files: str):
    if input_files[:len(S3_PREFIX)] == S3_PREFIX:
        yield from s3_iterator(input_files)
    else:
        yield from file_iterator(input_files)


def janky_labeled_jsonl_iterator(
    input_files: str,
    labels: List[str]
):
    for label, line in tqdm(input_iterator(args.input_files)):
        data = json.loads(line)
        text = " " + data['text']  # Prefix space for tokenizer.
        yield TOKENIZER.encode(text)
        labels.append(label)


def main(args):
    logger.info('Initializing vocabulary.')
    index = FMIndex()
    labels = []
    iterator = janky_labeled_jsonl_iterator(args.input_files, labels)
    logger.info('Building index...')
    index.initialize(iterator)
    index.labels = labels
    logger.info('Saving outputs.')
    index.save(args.output_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files', help='Input files.', type=str)
    parser.add_argument('-o', '--output_prefix', help='Output prefix.', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
