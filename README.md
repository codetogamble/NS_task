# NS_task
task for attempting fnc stance detection challenge

setup : 

install requirements.txt 
NOTE: tensorflow version is legacy version of 1.14.0 but the code has also been tested on version 2.3.1. Thus using updated tensorflow should not pose any issues.

download glove word embeddding from link : http://nlp.stanford.edu/data/glove.6B.zip
unzip the downloaded zip and move "glove.6B.50d.txt" to "embeddings" folder.




A version of model of type TRANSFORMER is hosted on GCP. To test results change body and headline accordingly and use the request below.

curl --location --request POST 'http://35.233.224.30:5000/query' --header 'Content-Type: application/json' --data-raw '{ "body":"Reddit got Hedgefunds for good, community power is strong. Spite of human nature can move large scale stock markets.", "headline":"Reddit can be strong." }'

sample result : {"result":{"agree":"0.54459983","disagree":"0.001165206","discuss":"0.06469633","unrelated":"0.38953862"}}
