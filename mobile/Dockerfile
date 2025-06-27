FROM node:18

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .

EXPOSE 19000
EXPOSE 19001
EXPOSE 8081

CMD ["npx", "expo", "start", "--tunnel", "--clear"]
