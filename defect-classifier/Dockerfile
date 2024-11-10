# Use an official PyTorch image as a base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "app.py"]
