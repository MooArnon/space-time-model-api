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
# To test lambda docker at local
`docker run --rm --env-file .env -p 9000:8080 etl`
`curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @price_data.json`
`curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations"`
