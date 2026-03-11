/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // Proxy /api-proxy/* → API service so the Docker image doesn't need the
  // external API URL baked in at build time.  The browser always calls the
  // rewrite path; Next.js forwards server-side using API_INTERNAL_BASE_URL.
  async rewrites() {
    const apiBase =
      process.env.API_INTERNAL_BASE_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api-proxy/:path*",
        destination: `${apiBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
