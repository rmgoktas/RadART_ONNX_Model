# RadART App ONNX Model
Model that predicts the style of paintings. Deployed on AWS EC2.
## How to test on my machine locally ?

- Clone the repository. Also you can download by ZIP.

- Update the value of the 'signature_dir' on line 17 in the 'app.py' file with the file path of the project.

- Install dependencies:
```bash
  pip install -r requirements.txt
```

- Run app.py:
```bash
  python app.py OR python3 app.py
```
- Create an API request to your localhost: 
```bash
  curl -X POST -H "Content-Type: application/json" -d '{"image": "$(base64 -w 0 YOUR_INPUT_IMAGE_PATH OR BASE64 FORMAT OF INPUT IMAGE)"}' http://127.0.0.1:5000/predict
```
- If it doesn't works, encode your input image to base64 from this website (https://www.base64-image.de) and paste after "image: " directly in request command.


## How to create an API endpoint ?

You can learn how to deploy this or a similar model on AWS and create an API endpoint from this article > 
https://medium.com/@rmgoktas/deploy-a-machine-learning-onnx-model-on-aws-ec2-using-flask-41d0823b7064

  
