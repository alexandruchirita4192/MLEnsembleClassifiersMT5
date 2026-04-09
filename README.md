# ML Ensemble MT5 strategy files

## Fisiere
- `train_mt5_ensemble_classifier.py`
- `MT5_Ensemble_Classifier_ONNX_Strategy.mq5`

## Instalare Python
```powershell
pip install MetaTrader5 pandas numpy scikit-learn lightgbm skl2onnx onnx
```

## Exemplu rulare
```powershell
python train_mt5_ensemble_classifier.py --symbol XAGUSD --timeframe M15 --bars 20000 --horizon-bars 8 --train-ratio 0.70 --output-dir output_ensemble_XAGUSD_M15_h8
```

## Output
Scriptul salveaza 3 modele ONNX:
- `mlp.onnx`
- `lightgbm.onnx`
- `hgb.onnx`

EA-ul face media probabilitatilor celor 3 modele.

## Pasii pentru MT5
1. Copiaza cele 3 fisiere ONNX langa `.mq5`
2. Recompileaza EA-ul
3. Ruleaza testerul doar pe `TEST UTC` din `run_in_mt5.txt`

## Setari de start recomandate in EA
- `InpUseTrendFilter = true`
- `InpTrendMAPeriod = 100`
- `InpUseTrendDistanceFilter = false`
- `InpUseAtrVolFilter = true`
- `InpAtrMinPercentile = 0.25`
- `InpAtrMaxPercentile = 0.85`
- `InpUseKillSwitch = false`
