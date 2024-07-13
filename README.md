# space-time-model-api
A constructor for prediction API

# To run
```bash
docker run --rm -it registry.digitalocean.com/space-time-image-registry/catboost ${feast_endpoint} ${entity}
```

# To deploy model
```bash
python deploy_model.py ${registry_endpoint} ${model_type}
```
python deploy_model.py registry.digitalocean.com/space-time-image-registry xgboost

'docker run --rm random_forest python classifier_model_api.py http://157.245.158.204:6000/feature/online_feature/fetch 11004'