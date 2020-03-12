# cifar-example-project

## Custom generation of Reports
### Prerequisites

* Login into your MiniO instance and make sure you have 3 buckets available:
    * models
    * reports
    * dataset

* Upload a *test.csv* validation data set into your **dataset** bucket.

* In Scaleout Studio, add a Model under Models and get the UID which has been assigned to that model 
(use our API for that).

* Upload a sequential model into your **models** bucket and make sure the name of the model is the UID assigned to the
model in Scaleout Studio.

* Change the endpoint, access_key and secret_key (in *default.py*) to be accurate for your Project's MiniO instance:

```python
minioClient = Minio('minio.generators.platform.demo.scaleout.se',
                    access_key='GXfzlBNHX2dALokXArRm',
                    secret_key='yI5pGGjgLBB62JOeCYMOHrsTOH9C2mYp1NhkB9r2',
                    secure=False)
```

* Create a new report generator for your Project under Reports in Scaleout Studio. Use the default generator 
*default.py* as a starting point.

* Go back to Models in Scaleout Studio, click Reports for a specific model and select the *default.py* as a generator.

* You are ready to generate your custom reports.
