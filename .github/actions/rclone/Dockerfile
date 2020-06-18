FROM alpine

RUN apk add --no-cache bash curl unzip ca-certificates fuse openssh-client \
  && wget -qO- https://rclone.org/install.sh | bash \
  && apk del bash curl unzip

ADD *.sh /

ENTRYPOINT ["/entrypoint.sh"]
