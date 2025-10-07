# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which reduces the image size.
# --trusted-host pypi.python.org: Can help avoid SSL issues in some network environments.
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container at /app
# This includes main.py and your model.pkl file
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run main.py when the container launches
# uvicorn main:app tells uvicorn to run the 'app' object from the 'main.py' file.
# --host 0.0.0.0 makes the app accessible from outside the container.
# --port 8000 specifies the port to run on.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
