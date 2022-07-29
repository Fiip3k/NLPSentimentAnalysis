# NLPSentimentAnalysis

ML Natural Language Processing web app classifying reviews as positive or negative. Using Python Flask to run the web app in a Docker container.


Pipelines:
1. Database data feed - saving raw data in a SQL database
2. Data preprocessing - preprocessing string data to match model input
3. Database preprocessed data feed - saving preprocessed data to another SQL table
4. Model training - using preprocessed data from the database to train several models, pick and save the best one
5. Output pipeline - Docker analyze() method - loads saved model to docker and uses it to analyze string sent in request

Some of them take a LONG time on a casual machine. For testing purposes I suggest using the uploaded trained model and vocabulary. 

Flask running inside Docker, successfully processing a POST request.
![docker_console](https://user-images.githubusercontent.com/29914639/181823989-8a34aa11-9cc8-49a6-b1f8-a9f28a259efa.PNG)

Sending POST request through Postman app and getting the desired answer from inside the Docker.
![docker_test](https://user-images.githubusercontent.com/29914639/181824028-2d674c85-6ccd-47c4-abbc-cfb2a5cad30f.PNG)
