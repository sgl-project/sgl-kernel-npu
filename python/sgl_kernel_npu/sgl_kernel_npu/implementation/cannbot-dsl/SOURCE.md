# DeepSeek V4.1 CANNBot-DSL source provenance

The seven Python files under `attention/` and `indexer/` come from pinned upstream samples.
[UPSTREAM.json](UPSTREAM.json) records their repository, source commit, original
paths, SHA-256 hashes, runtime versions and local backports.

They use the CANN Open Software License Agreement Version 2.0; see
[LICENSE.cannbot-dsl](LICENSE.cannbot-dsl).

The K projection backport pads backing storage to a multiple of 16 rows because
fixpipe can write a full tail tile. The pipeline must still receive the logical
T-row view: passing the padded tensor would change the postprocessing row count.
Keep the backing allocation and logical view together when updating this code.

Files follow the repository domain layout and isort/Black style. Local hashes
include formatting; upstream hashes remain available in the manifest.
