# space-time-model-api
A constructor for prediction API

# Run the flow
## Prediction mode
```bash
python classifier_model_api.py predict --asset ${asset_symbol} --evaluation-period 180
```

## Evaluation mode
```bash
python classifier_model_api.py evaluate --asset ${asset_symbol}
```

# Docker
## To run
```bash
docker run --rm -it registry.digitalocean.com/space-time-image-registry/catboost ${feast_endpoint} ${entity}
```

## To deploy model
```bash
python deploy_model.py ${registry_endpoint} ${model_type}
```


