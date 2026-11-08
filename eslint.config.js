import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["dist/**", "output/**", "experiments/**"],
  },
  js.configs.recommended,
  // 型情報を使うルールまで有効にする。計算的センサーは強いほど良い
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: {
          // tsconfig の include に入らない設定ファイル自身も型付きで検査する
          allowDefaultProject: ["eslint.config.js"],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
);
