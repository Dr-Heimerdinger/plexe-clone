# Multi-stage build for React frontend
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files (handle both npm and yarn)
COPY package.json ./
COPY package-lock.json* yarn.lock* ./

# Install dependencies (use npm install since package-lock.json may not exist)
RUN if [ -f yarn.lock ]; then yarn install --frozen-lockfile; else npm install; fi

# Copy source
COPY . .

# Build the app
RUN npm run build

# Final stage: serve with nginx
FROM nginx:alpine

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config to serve SPA (single-page app)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
