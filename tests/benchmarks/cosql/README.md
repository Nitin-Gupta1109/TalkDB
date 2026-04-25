# CoSQL benchmark data

This directory is where the CoSQL corpus goes. It's **gitignored** — don't commit the data.

## Layout we expect

```
tests/benchmarks/cosql/
├── README.md                        # this file
├── cosql_dataset/
│   ├── sql_state_tracking/
│   │   ├── cosql_dev.json           # ~293 dialogues, ~1300 turns (the dev split)
│   │   └── cosql_train.json         # we don't use this
│   └── database/
│       ├── academic/
│       │   └── academic.sqlite      # ~200 SQLite DBs, one per schema
│       ├── ...
│       └── world_1/
│           └── world_1.sqlite
└── tables.json                      # schema catalog for all 200 DBs
```

## Where to download

CoSQL is hosted by Yale LILY. Canonical source:

- **Landing page:** https://yale-lily.github.io/cosql
- **Data (Google Drive, ~100 MB):** see the "Data" link on the landing page — it redirects to a Drive folder

Yale's Drive links have historically been flaky. If the direct download breaks, two reliable mirrors:

- The Spider suite bundle on HuggingFace: `datasets/spider` and `datasets/cosql` (community uploads — verify schema matches the canonical format)
- The original `taoyds/cosql` GitHub repo README has a current link

## What to keep

For the dev-set eval we only need:

1. `sql_state_tracking/cosql_dev.json` (the questions + gold SQL + dialogue structure)
2. `database/*/` (the SQLite files for the DBs referenced in dev)
3. `tables.json` (schema metadata)

Train + test splits can be deleted to save space.

## Verifying the download

After unpacking, run from project root:

```bash
python -m tests.benchmarks.cosql_eval --verify
```

This walks every dialogue in dev, checks the referenced DB file exists, and reports any missing pieces without making any LLM calls.
