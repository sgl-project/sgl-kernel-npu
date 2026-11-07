# DeepSeek V4.1 CANNBot-DSL source provenance

The seven Python files under `attention/` are copied from the pinned upstream
samples, with the projection-tail allocation backport recorded in UPSTREAM.json. [UPSTREAM.json](UPSTREAM.json) records the repository, commit,
original paths, SHA-256 hashes and selected runtime versions.

They use the CANN Open Software License Agreement Version 2.0; see
[LICENSE.cannbot-dsl](LICENSE.cannbot-dsl).

K projection backing is rounded to 16 rows while the pipeline receives only
the logical T-row view. The backport changes allocation extent, not kernel math.
