import { dirname } from "path"
import { fileURLToPath } from "url"
import { FlatCompat } from "@eslint/eslintrc"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const compat = new FlatCompat({
  baseDirectory: __dirname,
})

const eslintConfig = [
  // Global ignores. This is a v0-generated codebase with existing issues, so we
  // ignore generated / vendored surfaces and keep the gate pragmatic.
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "out/**",
      "build/**",
      "coverage/**",
      "next-env.d.ts",
      // shadcn/ui primitives are vendored generated code.
      "components/ui/**",
      // owned by another concurrent wave; excluded from the lint gate for now.
      "lib/examples/**",
    ],
  },
  // Next.js recommended config + TypeScript support (parser + plugin ship with
  // eslint-config-next, so no manual parser registration is needed here).
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    rules: {
      // Warnings, not errors: surface issues without failing every dev run on a
      // pre-existing codebase. CI runs `--max-warnings=0` to still gate on them.
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // Too noisy for a v0 codebase that leans on `any` in places.
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-empty-object-type": "off",
      "react/no-unescaped-entities": "off",
      "react-hooks/exhaustive-deps": "warn",
      "@next/next/no-img-element": "off",
    },
  },
]

export default eslintConfig
