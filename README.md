# Digit Recognition Service


# Description
This web service host a digit recognition model, i.e. given a image containing a single digit, it will predict which of the 10 digits is in the image.

Logically, the program could been seen as two parts:
    
   * Model:
    The current hosted model are taken from [Keras example](https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py) which use a CNN trained on MNSIT to do the classification
    * Web Service:
    A Flask-based web service, exposing a JSON in/out REST API, which could be invoked by following HTTP request
```
        POST /recognize HTTP/1.1
        Host: example.com
        Content-Type: application/json
        {
         "image": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA8klEQVR4nGNgGKZAxaLz5PF4VWxSujNe/QOBX5ens6FJ6c388O/f42Wtv0/8e/owA1VuJlDX7j4OBob9Gse+3f8riiTFUff338sGbhDzkrYb0GhkSY/P/56YgRjMCiUPP/37v4gVSdL7w7/7uV2bNq289Ps5UN8LNWQbOdd/+fv/37/fYNf+WS2J5liBjsMbJs08AZKcJoA9FBb9+/cxmRm7XNmvf/+isEsxpHz69+8yO3Y5s4///n2yxqGx+d+/Lw445Hh//vs3A4ccz5N//y5w4JD0AwaCEw45hov//nXikmN4/P8leqghQOH/HJxy2AAAN2h6d/tZuR8AAAAASUVORK5CYII="
        }

#   the value of 'image' field in the body are base64 encoded image data
```

         
# Design and Trade-off
## The line between Web Service and Model
 Firstly, the handling of incoming JSON data and model related code should be separated,
 In Python ecosystem, Numpy ndarray is lingua franca in both image processing and machine learning,
      so I choose it as the data structure to cross boundary between web service and model object.
      This allows separate the two further in future, when needs arise.
     
## Model Class
In an abstract sense, one "model" can be different from another in terms of: <b>Machine Learning algorithm, traininig dataset, hyperparemeters, optimizers and its state</b>. Also, at the pre-processing step, the data can be transformed into different size/shape, color space, or some filters could be applied as well.  
Further, the software package used could different too, e.g. <b>sklearn, keras</b>.
      
Based on the above understanding, the web service needs to provide a flexible way to plugin/out different models, further, it might be a good idea if the very same model object could be directly imported when used in offline training AND serving.

So in this implementation, I choose to wrap native Keras model with a self-defined model class.
Two methods were provided ```pre_process``` and ```predict``` and should be implemented by different models
 
### Model Registry
A Model Registry and factory-like mechanism was implemented to load models from serialized model file and initialize model instance.

```
class MNISTKeras(object):
	# implementation skipped ...


MODEL_REGISTRY = {
    'MNISTKeras': MNISTKeras
}
```
Every time a new model is implemented and trained, a name should be given (perhaps in similar sense of "vgg16", "inception_v3") and used as key to loading that model.

### Model deployment:
* A configuration file is needed to specify which model and model file to be used as well as other possible environment parameters.
* Another idea for ease of deployment is store the model files in a object storage (e.g. S3) So when deploying new model, only need to change configuration and the application will fetch specified the model file and reload the model. The code for web service need not be touched.



## Web Service:
Following are considerations to make the service more production ready

### JSON input
The requirement stats that the API needs to be JSON in/out. One easy and quick way to upload binary data through JSON is encode the data in base64.
 
However this increase the data size by about 33% (pre-compressed) and adds overhead for decoding
As a reference, the [Google Drive API](https://developers.google.com/drive/v3/web/simple-upload) implement this in a more "native" way and should be regarded more RESTful.
            
### HTTP status semantic
The actual http status code should be semantically correct. For instance, "400 BAD REQUEST" should be returned as http status when encountering corrupted image data, or mandatory field is missing in request.
It seems that the Flask-JSON plugin I picked to assemble JSON response does not implement this way.

### Logging
Every incoming image should be logged and stored, so it could be used for further training.
        
### Image Size
In the wild, the web service will likely encounter large or larger than expeteced image.
        
### Off-loading
Image data transfer, decoding, transformation, prediction are major computations involved in this service. It will be a good idea to separate web service and prediction service into different processes running on separate machines.

            
## Others
Link to the hosted [Digit Recognition Service](http://ec2-35-158-228-11.eu-central-1.compute.amazonaws.com:8000/)

In this Github account you could also find my other Computer Vision related code



