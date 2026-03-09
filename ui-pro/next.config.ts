import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
  outputFileTracingExcludes: {
    '*': [
      'api/**/*',
      'data/**/*',
      'downloads/**/*'
    ]
  }
};

export default nextConfig;
