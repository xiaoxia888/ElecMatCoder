# frontend-next

独立的新前端原型，技术栈为：
- React
- Vite
- Tailwind CSS
- shadcn 风格的本地组件组织

## 运行

先安装依赖：

```bash
cd apps/platform/frontend-next
npm install
```

开发模式（默认代理到正式后端 `8000`）：

```bash
npm run dev
```

测试环境开发模式（代理到测试后端 `8001`）：

```bash
npm run dev:test
```

正式环境开发模式（代理到正式后端 `8000`）：

```bash
npm run dev:prod
```

构建：

```bash
npm run build:test
npm run build:prod
```

接口代理：
- 测试环境：`/api` -> `http://127.0.0.1:8001`
- 正式环境：`/api` -> `http://127.0.0.1:8000`

如后端端口不同，修改 `.env.test` 或 `.env.production` 中的 `VITE_API_PROXY_TARGET`。
