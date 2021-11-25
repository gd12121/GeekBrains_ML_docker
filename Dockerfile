FROM python:3.7
LABEL maintainer="gortunovd@inbox.ru"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/app/docker-entrypoint.sh"]