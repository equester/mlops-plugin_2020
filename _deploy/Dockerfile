FROM tiangolo/uvicorn-gunicorn:python3.6-alpine3.8

# Make directories suited to your application
RUN mkdir -p /home/project/app
WORKDIR /home/project/app

# Copy and install requirements
COPY requirements.txt /home/project/app
COPY . /home/project/app
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
# RUN pip --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org install /home/project/app/lib/numpy-1.18.4-cp38-cp38-manylinux1_x86_64.whl
RUN pip --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org install --upgrade pip
RUN pip --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org install --no-cache-dir -r requirements.txt --user
# Copy contents from your local to your docker container
