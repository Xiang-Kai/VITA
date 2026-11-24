# Contributing

[Documentation](index.md) / Contributing

Keep contributions small, testable and explicit about scientific scope.
Do not describe engineering smoke results as paper reproduction.

## Before A Change

1. Read the [methodology](methodology.md) and [training contracts](training.md).
2. Use a new branch and an isolated environment.
3. Preserve existing runs and failures. Never rewrite provenance or expected
   results simply to make an audit pass.
4. Do not run real robots, external model services or shared GPU jobs as part
   of ordinary unit tests.

## Local Checks

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
python scripts/check_docs.py
python scripts/build_release.py
git diff --check
```

Regression tests should generate clearly synthetic inputs in temporary
directories. Simulator receipts used as mocks must carry `synthetic_fixture`.
New failure handling must preserve the original error and must not turn a
failed operation into successful evidence.

## Compatibility

Prepared plans and checkpoints record hashes of every `vita/*.py` source file.
Changing runtime source can invalidate exact resume and cross-environment
evaluation. Describe compatibility impact in a pull request; do not remove
hash checks or modify old checkpoints to disguise a new experiment.

Documentation/presentation changes alone should leave the training schedules
and source hashes unchanged. Test example commands against the actual CLI.

## Public Files

Add new publishable files to [`.release-files`](../.release-files). The source
builder reads only that allowlist. Keep machine paths, credentials, datasets,
weights, videos and run logs out of source commits. Sanitized examples must
not contain real deployment details.

Report dependency versions and concise reproduction steps in issues, rather
than uploading entire runtime directories. Review the [release policy](releasing.md)
before sharing model artifacts or detailed logs.

## Pull Requests

Explain the problem, the bounded change, the tests performed and any remaining
validation limits. Include a regression case when fixing a defect. For changes
to an algorithm or dataset protocol, state how the new behavior differs from
the paper and from the existing independent implementation.
