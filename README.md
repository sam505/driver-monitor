# Guardian Driver

Due to the high number of accidents that happen annually, I decided to come up with a computer vision
project that aims at tracking the driver's attention at any time the vehicle s on the road. This is achieved by
estimating the head angles and the directions of the driver's eyes.
In case the head angles are beyond the allowed range, the driver is alerted before losing control of the vehicle.

# Run The Project
I am assuming that you have downloaded the compressed deployment package. Extract the files
and navigate to the deployment directory. Navigate in the project folder named `Guardian_driver`. Create a virtual environment 
using Virtualenv with the command below.
```
virtualenv venv
```
Source the virtual environment.
```
source venv/bin/activate
```
Using pip install the packages listed in the `requirements.txt` file required in the project.
```
pip install -r requirements.txt
``` 

Navigate back to the main folder `deployment`, you will see a setup.sh file.

Use the below command to setup the OpenVINO Toolkit environment variables the source then before running the main app
 for the first time.
 
 ```
 bash setup.sh
 ```

After installing all the dependencies needed, navigate to the project folder `Guardian_driver`.
Inside the directory there is s file names `run_project.sh`. Use the command below to run the project.
```
bash run_project.sh
```

When running the project and it is not the first time, there is no need of installing all the dependencies again. While in the
deployment directory, run the `source_environment.sh` file using the command below to source for the OpenVINO Toolkit environment variables.
```
bash source_environment.sh
```
Afterwards navigate to the project folder named `Guardian_driver` and run the `run_project.sh` file using the command below.
```
bash run_project.sh
```