# Phase 3 results archive

Before cleanup, raw results and historical reports were archived and every
member was read back and verified against its SHA-256 and byte size. The archive
is private, outside the working tree at `~/GarageRAG-archives/phase3-20261026/`.
An identical local mirror is at ignored `output/archives/phase3-20261026/`.
Keep the external directory in private backups; a fresh Git clone does not contain it.
Do not upload the archive: it includes private passages, requests and captures.

- Source HEAD: `25cacb07f184915a8ddb28ff5da9b1fc5b6fb1a0` (working snapshot before documentation cleanup).
- `results.tar.gz`: SHA-256 `dfa5ff79cbc3f857a006db1ff3d97f4e9fa1c14f1b525135d8a2dc311f400fed`.
- `MANIFEST.json`: SHA-256 `1138be6b3dbe5b3f7faf618268a0818e0b16689badb6d5f2db931f3d741f8771`.
- 285 members / 28,577,614 uncompressed bytes.

The full manifest lists every relative path, byte count and SHA-256. Preserved
files include original public results and reports, all pre-cleanup local output
(checkpoints, private drafts/contexts/reviews and timing), datasets, index
manifests, experiment runners and the zero-API repair replay script. Original
parameters, runtime source hashes, corpus/index identities and provenance remain
inside those unmodified files. Credentials, `.env` and model weights are excluded.
The original private PDF remains in its existing local corpus location; obtain
it privately and verify its recorded corpus SHA when reconstructing an index.

## Verify and restore

From the archive directory:

```bash
shasum -a 256 -c SHA256SUMS
mkdir -p /tmp/garage-rag-phase3-restore
# Use a fresh empty private directory, never overwrite the active checkout.
tar -xzf results.tar.gz -C /tmp/garage-rag-phase3-restore
```

Verify restored members from the same archive directory:

```bash
python3 - <<'VERIFY'
import hashlib, json
from pathlib import Path
root = Path('/tmp/garage-rag-phase3-restore')
for entry in json.loads(Path('MANIFEST.json').read_text())['files']:
    data = (root / entry['path']).read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
print('All members verified')
VERIFY
```

`checkout/` restores the snapshotted paths; `local-runners/` contains historical
experiment runners and `phase3-offline-replay.py`. Check each restored file
against the manifest's `files[].sha256` and `files[].bytes` before use. Run the
repair replay with the current checkout's `.venv/bin/python`, from the checkout,
after restoring private captures under ignored `output/private/`. Matching
cached E5 weights and the private existing index are needed for its local search;
it makes no API calls. Preserve old checkpoints; use new names for future runs.

## Removed public artifacts

Historical observations are summarized in [answer evaluation](grounded-answering.md).
Hashes below identify the exact archived bytes; the full payloads add no runtime
or test dependency. All 43 paths in the original PR were audited: retain runtime
modules and their tests, synthetic dataset, configuration/package files and
current architecture/usage/testing docs; consolidate experiment history here and
in the single evaluation summary. Frozen Phase 2 files are untouched.

| Original benchmark artifact | Lines | SHA-256 |
| --- | ---: | --- |
| `groq-20261026-cold-pressure-de-review.v1.json` | 30 | `222d5b0cc9dc68fc15ae728e718689c662b158d63ab64311470ed6abaed471bd` |
| `groq-20261026-cold-pressure-de-ru-timing.v1.json` | 164 | `f35c047853355841c61e44c0f0d85099a4be14b34abb468991f4bc0ab2645ae4` |
| `groq-20261026-cold-pressure-de.v1.json` | 480 | `a2b1bc2ac017f80aadc02a1ca9f002b90c669e4fcef8301e77b9e34dea199682` |
| `groq-20261026-cold-pressure-en-review.v1.json` | 30 | `86754616efa5525cac00248d7fb6c54afff70013c0fcbe231fb81df47c440d96` |
| `groq-20261026-cold-pressure-en.v1.json` | 460 | `b4b30e35af3a3180a41aa0aaf9b1c93b59e3a965d60f5ef136c654e0829de928` |
| `groq-20261026-cold-pressure-multilingual.v1.json` | 331 | `85dd2df3c023cfad57d2880ccdb7b264511fe2a2a7070b70f88916ee67422321` |
| `groq-20261026-cold-pressure-ru-review.v1.json` | 30 | `4967f49f2ca7289f63a6bbd4cbfc2fb4d6be654b83ebd642087bf21611618f47` |
| `groq-20261026-cold-pressure-ru.v1.json` | 492 | `07070e0d6f504e0bafec16055600fec76629a1af1e0e9acf65933d9282944d12` |
| `groq-20261026-e39-results.v1.json` | 7523 | `aea807340040d6133dcd40ff580d1dcd935d9e287416af27e86ab1aa3c79fcb5` |
| `groq-20261026-hard-en-results.v1.json` | 2484 | `184c736f46d4c34af038caaa039230ba87809ed7158fcc77b94e89e4fbc8422c` |
| `groq-20261026-hard-en-review.v1.json` | 155 | `414f87ae0c8923e863eb44f78a6da1550099db6e3dbbcf9e431d1f647eed9edf` |
| `groq-20261026-hard-en-timing.v1.json` | 405 | `de3c5f19b276dda9d47d9d422a328cffdd171ebbe2293ff762b892a508dfd26f` |
| `groq-20261026-smoke.v1.json` | 108 | `2e76225d630d9862db43594adfc8384bc3554082f90ec908bfdc856355665f65` |
| `groq-e39-20261026.md` | 305 | `403abb58133f8540250f9236fb5d42acdc484384edaa14ee8e71f3ad5cfd4a0c` |
| `groq-hard-en-20261026.md` | 199 | `2ddaf484bfbed4d365072b1a2a0c14d8aec169e5dbcfe28b1bde907a4eac15fa` |
| `phase3-20261026-results.v1.json` | 829 | `574f3763020e5c632fbcaf2a4acf9848b222a93a35b4ec4f306b663c3560f262` |
| `phase3-results.v1.json` | 16323 | `a723279d43fc59391961e9f1a605156dbacb61d242b279e68bf0477f796a4567` |
