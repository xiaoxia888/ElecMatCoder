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

开发模式：

```bash
npm run dev
```

默认端口：`5174`

接口代理：
- `/api` -> `http://127.0.0.1:8000`

如后端端口不同，修改 `vite.config.ts`。
