# NS_task serve_files

This folder contains copy of previous section in order to deploy to a live server of Google cloud or aws.
To run locally, run the following commands from this folder path

export FLASK_APP=server_1.py
flask run

curl --location --request POST 'localhost:5000/query' \
--header 'Content-Type: application/json' \
--data-raw '{
    "body":"Reddit got Hedgefunds for good, community power is strong. Spite of human nature can move large scale stock markets.",
    "headline":"Reddit can be strong."
}'

