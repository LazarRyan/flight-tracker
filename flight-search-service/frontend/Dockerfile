# Use the official Nginx image as a parent image
FROM nginx:alpine

# Copy the static content to Nginx's default public directory
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start Nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]
