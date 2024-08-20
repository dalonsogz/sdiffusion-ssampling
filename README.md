# GUI App with Stable Diffusion, Docker and Flask - Beginner Friendly!
A Flask web GUI application that generates and upscales images from text prompts with Stable Diffusion and EDSR.

## Introduction

Learning application about how to use Generative AI application with **Stable Diffusion**, **Docker**, and **Flask**. 
The app will take in a user-provided text prompt and convert it into high-resolution images (up to 2048 x 2048 pixels!) with EDSR in TensorFlow.

- Other technologies and kwnoledge involved: Docker Init, Diffusers, CUDA, FreeU, OpenCV, and Docker Scout.

- The source site of the the tutorial to build application: https://www.pythonsimplified.org/post/create-text-to-image-gui-app

## Building and running

- At 'starter_files' directory: docker compose up --build
- Application will be available at <a href="http://localhost:8080" target="_blank">localhost:8080</a>

## Screenshot

<img src="./starter_files/static/images/screenshot.png" width=600px>

## Useful links

- The video tutorial: https://www.youtube.com/watch?v=gDF_qGzYEYQ
- Files required to implement the project mostly from scratch <a href="https://github.com/MariyaSha/StableDiffusion_GUI_App" target="_blank">video tutorial</a>

### Main links

- Stable Diffusion model: https://huggingface.co/CompVis/stable-diffusion-v1-4
- EDSR in Tensorflow: https://github.com/Saafke/EDSR_Tensorflow/tree/master

### Other links

- FreeU: Free Lunch in Diffusion U-Net() for improving quality results: https://github.com/ChenyangSi/FreeU
- NVIDIA Container Toolkit for CUDA support:https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Turn on GPU access with Docker Compose: https://docs.docker.com/compose/gpu-support/

## Docker: build and deploy

### Building and running your application

When you're ready, start your application by running:
`docker compose up --build`.

Your application will be available at http://localhost:8000.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)

## Credits

- Create GUI App with Stable Diffusion, Docker, and Flask - Beginner Friendly!: https://www.pythonsimplified.org/