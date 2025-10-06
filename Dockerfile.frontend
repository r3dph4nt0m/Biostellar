FROM node:18-alpine

WORKDIR /app

COPY backend/ai/frontend/package*.json ./
RUN npm install

COPY backend/ai/frontend ./

EXPOSE 3000
CMD ["npm", "start"]
