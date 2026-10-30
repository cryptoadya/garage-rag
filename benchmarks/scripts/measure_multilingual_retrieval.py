"""One fresh-process local retrieval measurement; frozen translations, no provider calls.

Run once per variant/corpus/repetition. Use an extracted main/src for --source-dir
with --variant main, and this checkout's src with --variant pr. Models are CPU-only;
weight files must already be cached. Warm calls deliberately do not cache vectors
or reranker predictions. Output contains timings/counts, never source passages.
"""

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--variant', choices=('main', 'pr'), required=True)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True,
                        help='JSON object mapping en/de/ru to frozen parallel questions')
    parser.add_argument('--warm-runs', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error('--warm-runs must be positive')
    questions = json.loads(args.questions.read_text())
    sys.path.insert(0, str(args.source_dir.resolve()))
    started = time.perf_counter()
    from garage_rag.embeddings import E5Embeddings
    from garage_rag.index import Index

    stages = {'reranker_load_ms': 0., 'reranker_predict_ms': [], 'pairs': []}
    if args.variant == 'pr':
        import garage_rag.retrieval as retrieval

        class MeasuredReranker(retrieval.LocalReranker):
            def __init__(self):
                tick = time.perf_counter()
                super().__init__()
                stages['reranker_load_ms'] = (time.perf_counter() - tick) * 1000

            def predict(self, pairs, **kwargs):
                tick = time.perf_counter()
                values = super().predict(pairs, **kwargs)
                stages['reranker_predict_ms'].append((time.perf_counter() - tick) * 1000)
                stages['pairs'].append(len(pairs))
                return values

        retrieval.LocalReranker = MeasuredReranker

    with Index(args.index_dir) as index:
        manifest = index.manifest_path.read_bytes()
        embeddings = E5Embeddings()
        search = retrieval.AnswerSearch(index, embeddings) if args.variant == 'pr' else None
        peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tick = time.perf_counter()
        languages = search.source_languages if search is not None else ()
        language_ms = (time.perf_counter() - tick) * 1000
        # main translates RU/DE into EN; PR keeps RU and adds missing languages.
        queries = (questions['ru'], *(questions[lang] for lang in languages if lang != 'ru'))

        def retrieve():
            return (search(queries, 5) if search is not None else
                    index.search(questions['en'], embeddings, 5))

        tick = time.perf_counter()
        hits = retrieve()
        cold_search_ms = (time.perf_counter() - tick) * 1000
        cold_total_ms = (time.perf_counter() - started) * 1000
        peak_cold = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        warm = []
        for _ in range(args.warm_runs):
            tick = time.perf_counter()
            retrieve()
            warm.append((time.perf_counter() - tick) * 1000)
        peak_warm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert manifest == index.manifest_path.read_bytes()
        import torch

        # macOS reports ru_maxrss in bytes; Linux reports KiB.
        divisor = 1024 ** 2 if sys.platform == 'darwin' else 1024
        result = {
            'variant': args.variant, 'platform': platform.platform(), 'device': 'cpu',
            'python': platform.python_version(), 'torch_threads': torch.get_num_threads(),
            'index_config': index.config.as_dict(),
            'chunks': index.stats()['chunks'], 'source_languages': languages,
            'queries': len(queries) if search is not None else 1,
            'source_language_ms': language_ms, 'cold_search_ms': cold_search_ms,
            'cold_total_ms': cold_total_ms, 'warm_ms': warm,
            'warm_median_ms': statistics.median(warm), 'warm_max_ms': max(warm),
            'process_peak_mib_before_search': peak_before / divisor,
            'process_peak_mib_cold': peak_cold / divisor,
            'process_peak_mib_warm': peak_warm / divisor,
            'reranker': stages, 'hits': len(hits), 'provider_calls': 0,
        }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
