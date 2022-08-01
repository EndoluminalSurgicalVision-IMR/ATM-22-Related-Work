# The Docker Tutorial for the ATM_22 Test Phase Submission
<div align=center><img src="../figs/bannar.png"></div>

## 1. Build a Docker Image
The configuration of the docker image should be correctly set in `DockerFile`, and then you could build the docker image with your teamname.
```angular2html
docker build -t teamname .
```

## 2. Save the Docker Image
Please save the docker as belows, and e-mail to <a href="mailto:IMR-ATM22@outlook.com">IMR-ATM22@outlook.com</a> with a downloadable link, along with a short paper that describes your methods and experiment settings.
```angular2html
docker save teamname:latest -o teamname.tar.gz
```












