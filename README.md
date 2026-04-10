# Weighted ML ensemble classifiers for MT5

Collection of 3 best individual classifiers: MLP, LightGBM, HGB.

## Files
- `train_mt5_weighted_ensemble_classifier.py`
- `MT5_Weighted_Ensemble_ONNX_Strategy.mq5`

## What it brings extra
You can switch from inputs / arguments:
- from `MLP only`
- to `LightGBM only`
- to `HGB only`
- or to any intermediate combination

Examples:
- `MLP only`: `--mlp-weight 1 --lgbm-weight 0 --hgb-weight 0`
- `LightGBM only`: `--mlp-weight 0 --lgbm-weight 1 --hgb-weight 0`
- `HGB only`: `--mlp-weight 0 --lgbm-weight 0 --hgb-weight 1`
- `Weighted`: `--mlp-weight 0.6 --lgbm-weight 0.25 --hgb-weight 0.15`

The script automatically normalizes the weights.

## Python Installation
```powershell
pip install MetaTrader5 pandas numpy scikit-learn lightgbm skl2onnx onnxmltools onnx
```

## Example run learning without specifying weights, searching for the optimal weight combination
```powershell
python train_mt5_optimize_ensemble_weights.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --weight-step 0.2 --allow-zero-weights --output-dir output_weight_search_h8_fast
```

## Example run learning with specified weights (only MLP according to the example weights)
```powershell
python train_mt5_weighted_ensemble_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 1 --lgbm-weight 0 --hgb-weight 0 --output-dir output_weighted_mlp_only_XAGUSD_M15_h8
```

## Important for MT5
Put next to EA all 3 files:
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`

Even if you use a single model, EA can be changed only from inputs, without recompilation.
