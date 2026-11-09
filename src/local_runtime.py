"""Local OpenAI-compatible runtime and BIRD export helpers."""

import json
import math
import os
from pathlib import Path


def api_base(value: str) -> str:
    """Accept either a server root or its /v1 base URL."""
    value = value.rstrip('/')
    return value if value.endswith('/v1') else value + '/v1'


def add_local_arguments(parser) -> None:
    parser.add_argument('--local_model', default=os.getenv('CHESS_LOCAL_MODEL'))
    parser.add_argument('--base_url', default=os.getenv('OPENAI_API_BASE'))
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--embedding_base_url', default=os.getenv('CHESS_EMBEDDING_BASE_URL'))
    parser.add_argument('--embedding_model', default=os.getenv('CHESS_EMBEDDING_MODEL'))
    parser.add_argument('--db_root_directory', help='Directory containing the database folders')
    parser.add_argument('--output_dir')
    parser.add_argument('--resume', action='store_true',
                        help='Resume output_dir using its saved configuration')


def configure_local_runtime(args, parser) -> None:
    """Configure before importing modules that construct model clients."""
    if not args.local_model:
        return
    for name in ('base_url', 'embedding_base_url', 'embedding_model', 'db_root_directory'):
        if not getattr(args, name):
            parser.error(f'--{name} is required with --local_model')
    if args.temperature is None:
        args.temperature = float(os.getenv('CHESS_TEMPERATURE', '0'))
    if args.max_tokens is None:
        args.max_tokens = int(os.getenv('CHESS_MAX_TOKENS', '16384'))
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error('--temperature must be finite and nonnegative')
    if args.max_tokens <= 0 or args.num_workers <= 0:
        parser.error('--max_tokens and --num_workers must be positive')
    db_root = Path(args.db_root_directory).resolve()
    if db_root.name != f'{args.data_mode}_databases':
        parser.error(f'--db_root_directory must end in {args.data_mode}_databases')
    if not db_root.is_dir():
        parser.error(f'Database directory does not exist: {db_root}')
    args.db_root_directory = str(db_root)
    args.base_url = api_base(args.base_url)
    args.embedding_base_url = api_base(args.embedding_base_url)
    os.environ.update({
        'CHESS_LOCAL_MODEL': args.local_model,
        'CHESS_BASE_URL': args.base_url,
        'CHESS_TEMPERATURE': str(args.temperature),
        'CHESS_MAX_TOKENS': str(args.max_tokens),
        'CHESS_EMBEDDING_BASE_URL': args.embedding_base_url,
        'CHESS_EMBEDDING_MODEL': args.embedding_model,
        'DB_ROOT_PATH': str(db_root.parent),
    })
    os.environ.setdefault('INDEX_SERVER_HOST', 'localhost')
    os.environ.setdefault('INDEX_SERVER_PORT', '12345')


def local_chat_params() -> dict | None:
    if not os.getenv('CHESS_LOCAL_MODEL'):
        return None
    return {
        'model': os.environ['CHESS_LOCAL_MODEL'],
        'openai_api_base': api_base(os.environ['CHESS_BASE_URL']),
        'openai_api_key': os.getenv('OPENAI_API_KEY') or 'EMPTY',
        'temperature': float(os.environ.get('CHESS_TEMPERATURE', '0')),
        'max_tokens': int(os.environ.get('CHESS_MAX_TOKENS', '16384')),
        'timeout': float(os.environ.get('CHESS_REQUEST_TIMEOUT', '120')),
        'max_retries': int(os.environ.get('CHESS_MAX_RETRIES', '2')),
    }


def make_embeddings(default_model: str):
    from langchain_openai import OpenAIEmbeddings

    if not os.getenv('CHESS_EMBEDDING_MODEL'):
        return OpenAIEmbeddings(model=default_model)
    return OpenAIEmbeddings(
        model=os.environ['CHESS_EMBEDDING_MODEL'],
        openai_api_base=api_base(os.environ['CHESS_EMBEDDING_BASE_URL']),
        openai_api_key=os.getenv('CHESS_EMBEDDING_API_KEY') or 'EMPTY',
        check_embedding_ctx_length=False,
        chunk_size=4,
    )


def ensure_indexes(dataset: list[dict], db_root_directory: str) -> None:
    """Build missing indexes serially; never replace an existing vector store."""
    from database_utils.db_values.preprocess import make_db_lsh
    from database_utils.db_catalog.preprocess import make_db_context_vec_db

    identity = {'model': os.environ['CHESS_EMBEDDING_MODEL'], 'input': 'text'}
    for db_id in dict.fromkeys(row['db_id'] for row in dataset):
        directory = Path(db_root_directory) / db_id
        if not (directory / f'{db_id}.sqlite').is_file():
            raise FileNotFoundError(f'Missing database: {directory}')
        vector = directory / 'context_vector_db'
        marker = vector / 'chess_embedding.json'
        if vector.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != identity:
                raise ValueError(f'Unknown or incompatible embedding index: {vector}. '
                                 'Move it aside explicitly before rebuilding.')
            if not (vector / 'chroma.sqlite3').is_file():
                raise ValueError(f'Incomplete vector index: {vector}; move it aside before retrying')
        else:
            print(f'Building context index: {db_id}', flush=True)
            make_db_context_vec_db(str(directory), use_value_description=True)
            marker.write_text(json.dumps(identity, indent=2) + '\n')
        preprocessed = directory / 'preprocessed'
        if not all((preprocessed / f'{db_id}_{suffix}.pkl').is_file()
                   for suffix in ('lsh', 'minhashes')):
            print(f'Building LSH index: {db_id}', flush=True)
            make_db_lsh(str(directory), signature_size=100, n_gram=3,
                        threshold=0.01, verbose=True)


def export_bird_predictions(dataset: list[dict], result_directory: str) -> Path:
    """Export in dataset order, including failed or unfinished questions."""
    directory = Path(result_directory)
    predictions = json.loads((directory / '-predictions.json').read_text())
    ordered = {}
    delimiter = '\t----- bird -----\t'
    for index, row in enumerate(dataset):
        value = predictions.get(str(row.get('question_id', index)))
        sql = value.split(delimiter)[0].strip() if isinstance(value, str) else ''
        ordered[str(index)] = (sql or '-- generation failed') + delimiter + row['db_id']
    target = directory / f'{directory.name}.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(f'BIRD predictions: {target}', flush=True)
    return target
