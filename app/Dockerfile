# Using lightweight alpine image
FROM frolvlad/alpine-python-machinelearning

# Installing packages
RUN apk update

# Defining working directory and adding source code
WORKDIR /usr/src/app
COPY bootstrap.sh ./

# Install API dependencies
RUN pip install flask
RUN pip install flask-api

# Add application code.
COPY . ./

# Start app
EXPOSE 5000
ENTRYPOINT ["/usr/src/app/bootstrap.sh"]