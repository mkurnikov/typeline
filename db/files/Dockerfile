FROM postgres:latest

ENV LANG C.UTF-8

# see README for the postgres image for explanation
RUN mkdir -p /docker-entrypoint-initdb.d
COPY init.sh /docker-entrypoint-initdb.d/

COPY pg_hba.conf /etc/
COPY postgresql.conf /etc/

CMD ["postgres", "-c", "config_file=/etc/postgresql.conf", "-c", "hba_file=/etc/pg_hba.conf"]