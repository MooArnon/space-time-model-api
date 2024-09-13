sudo docker build -f catboost_model.Dockerfile --build-arg MODEL_TYPE=catboost -t space-time/classifier-model:catboost .
docker run --rm --env-file .env -p 9000:8080 space-time/classifier-model:catboost
docker tag space-time/classifier-model:catboost 888577051220.dkr.ecr.ap-southeast-1.amazonaws.com/space-time/classifier-model:catboost
docker push 888577051220.dkr.ecr.ap-southeast-1.amazonaws.com/space-time/classifier-model:catboost