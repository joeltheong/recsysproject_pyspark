# Include where we get the image from (operating system)
FROM python:3.8

#MAINTAINER Joel Ong 'e0685733@gmail.com'

# We cannot press Y so we do it automatically 
RUN apt-get update && apt-get install -y \
    git \
    default-jdk -y \
    curl \    
    ca-certificates \
    python3 \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

# Set working directory 
WORKDIR /app 

# Copy everything in currect directory into the app directory.
ADD . /app

# Install all of the requirements 
RUN pip install -r requirements.txt

# Environment 
ENV PYSPARK_PYTHON=python3

# CMD executes once the container is started
ENTRYPOINT ["streamlit","run", "streamlit.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD ["streamlit.py"]

    

