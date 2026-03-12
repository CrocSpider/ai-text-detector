# ---- deps ----
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- builder ----
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Ensure the public directory exists (Next.js standalone COPY requires it)
RUN mkdir -p public

# API_INTERNAL_BASE_URL is read at Next.js startup (by rewrites()) not baked
# into the client bundle, so it's safe to set here as a build default.
ARG API_INTERNAL_BASE_URL=http://ai-text-detector-api:8000
ENV API_INTERNAL_BASE_URL=$API_INTERNAL_BASE_URL

RUN npm run build

# ---- runner ----
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# standalone output bundles only the files Next.js actually needs
COPY --from=builder --chown=appuser:appgroup /app/public ./public
COPY --from=builder --chown=appuser:appgroup /app/.next/standalone ./
COPY --from=builder --chown=appuser:appgroup /app/.next/static ./.next/static

USER appuser

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
