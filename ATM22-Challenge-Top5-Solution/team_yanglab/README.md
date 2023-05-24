## Reproduce the work in the paper
The file format must be *.nii.gz  
1. Download the docker file through (no available at the moment due to AIIB23)
```https://drive.google.com/file/d/1K3JZsEOVBYX1QCnhNhrW6xOFwcZJrlwu/view?usp=sharing```  
2. Use the docker file for prediction  
```docker image load < yang.tar.gz```  
```docker container run --gpus "device=0" --name yang --rm -v "your_test_data_path":/workspace/inputs/ -v "your_test_output_path":/workspace/outputs/ yang:latest /bin/bash -c "sh predict.sh" ```
## Evaluation metrics
Please find the evaluation metrics from ```utils/airway_metric.py```
