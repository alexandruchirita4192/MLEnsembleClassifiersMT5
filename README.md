# Weighted ML ensemble classifiers for MT5

## Fisiere
- `train_mt5_weighted_ensemble_classifier.py`
- `MT5_Weighted_Ensemble_ONNX_Strategy.mq5`

## Ce aduce in plus
Poti trece din inputuri / argumente:
- de la `MLP only`
- la `LightGBM only`
- la `HGB only`
- sau la orice combinatie intermediara

Exemple:
- `MLP only`: `--mlp-weight 1 --lgbm-weight 0 --hgb-weight 0`
- `LightGBM only`: `--mlp-weight 0 --lgbm-weight 1 --hgb-weight 0`
- `HGB only`: `--mlp-weight 0 --lgbm-weight 0 --hgb-weight 1`
- `Weighted`: `--mlp-weight 0.6 --lgbm-weight 0.25 --hgb-weight 0.15`

Scriptul normalizeaza automat weight-urile.

## Instalare Python
```powershell
pip install MetaTrader5 pandas numpy scikit-learn lightgbm skl2onnx onnxmltools onnx
```

## Exemplu rulare invatare fara a specifica weight-urile, cautand combinatia optima de weight-uri
```powershell
python train_mt5_optimize_ensemble_weights.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --weight-step 0.2 --allow-zero-weights --output-dir output_weight_search_h8_fast
```

## Exemplu rulare invatare cu weight specificati (doar MLP conform weight-urilor puse ca exemplu)
```powershell
python train_mt5_weighted_ensemble_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --mlp-weight 1 --lgbm-weight 0 --hgb-weight 0 --output-dir output_weighted_mlp_only_XAGUSD_M15_h8
```

## Important pentru MT5
Pune langa EA toate cele 3 fisiere:
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`

Chiar daca folosesti un singur model, EA poate fi schimbat doar din inputuri, fara recompilare.
