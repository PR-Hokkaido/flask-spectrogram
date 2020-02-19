#Use the Python image from DockerHub as a base image
FROM python:3-slim

#Expose the port for your python app
EXPOSE 5000

#Copy all app files from the current directory into the app
#directory in your container. Set the app directory
#as the working directory
WORKDIR /flask-spectrogram/
COPY .  .

#Install any requirements that are defined
RUN pip install --no-cache-dir -r requirements.txt

#Update the openssl package
RUN apt-get update && apt-get install -y \
openssl libsndfile1-dev curl

#Start the app.
CMD ["python", "server.py"]

