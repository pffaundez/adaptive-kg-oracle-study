# Data

- `examples/`: tiny synthetic logs used for smoke tests.
- `raw/`: downloaded source datasets; never commit them.
- `processed/`: normalized questions and gold annotations; never commit large
  or restricted artifacts.
- `manifests/`: provenance, licenses, checksums, versions, and split definitions.

Each normalized example should eventually contain a stable question id, natural
language question, target KG/endpoint, gold SPARQL when available, gold answers,
split, language, and optional question-type annotations.

