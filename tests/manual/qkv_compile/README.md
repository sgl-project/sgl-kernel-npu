# Existing QKV compile-boundary evidence

`probe_qkv_custom_compile.py` is the existing generic probe. It compares original eager, custom eager and compiled execution, checks inputs are unchanged and runs schema/fake/AOT registration checks. Run it with explicit `--original`, `--candidate`, `--inputs`, `--output`, `--device`, `--input-kind` and `--input-note` arguments; the inputs file is a torch.save list of kwargs dictionaries and must contain at least two shapes. No local deployment path is hardcoded.

The saved synthetic test used seed 19, BF16, q=2048/kv=1024/head=128 and batches 1/2/16. Its original input archive SHA-256 is 6873e87935c387a8697320f23910c46e3161f1fcd291a277858eb4a528c4007e. The tensor archive is not present in this small source bundle; do not claim this bundle includes that exact historical fixture.

`qkv_compile_historical_report.json` is the early namespace-isolated wrapper report (candidate SHA e0e1d617fe75546b11e4bff72a7d9b14d0738e671c34e9986a53d91eb279f220), not the final file SHA. It records exact parity and opcheck results. `qkv_final_same_graph_report.json` separately identifies the final product file SHA a4ac7b8d5a5d5087a992654d473c565b3c974c12289aa3ecbc087090659d5971 and preserves same-capture input 1 -> 0.5 -> 1, exact output and no-alias results. Only deployment path fields were removed from public evidence copies; original files remain in the delivery archive.

The project did not rerun NPU experiments for publication. These fixed-shape results do not establish all parameter combinations, training, arbitrary layouts, or the main-branch non-power-of-two extension introduced later by #791.
